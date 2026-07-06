import json
import logging
import time
from datetime import date, datetime

from pydantic import ValidationError

from app.analysis.categories import CATEGORY_STAGE2
from app.analysis.fallback import fallback_analysis
from app.analysis.prompt import (
    ALLOWED_CATEGORIES,
    UNKNOWN_CATEGORY,
    LlmAnalysis,
    build_classify_prompt,
)
from app.config import Settings, settings
from app.integrations.kakao_local import KakaoLocalClient, enrich_restaurant_details_with_kakao
from app.llm.base import LlmClient, LlmError
from app.llm.factory import build_llm_client
from app.models import AnalyzeResponse

logger = logging.getLogger("capturemate.analysis")


class AnalysisService:
    """마스킹된 텍스트를 2단계로 분석한다.

    1단계(분류): 텍스트 -> 카테고리 + 어느 카테고리든 쓸 수 있는 기본 설명.
    2단계(세부 추출): 카테고리에 등록된 전용 프롬프트로 그 카테고리의 세부 필드 추출.
      - 카테고리별 프롬프트/모델은 app/analysis/categories/ 아래에서 담당자가 소유.
      - 2단계가 없는(미등록) 카테고리는 1단계 결과만 반환.

    신뢰성: 각 단계마다 JSON 요청 -> 파싱 -> Pydantic 검증 -> 백오프 재시도.
    1단계 실패 시 키워드 폴백으로 대체하고, 2단계 실패는 details=None으로 흘려보내
    전체 요청은 절대 완전히 실패하지 않는다.
    """

    def __init__(self, client: LlmClient | None = None, config: Settings = settings) -> None:
        self._config = config
        # 시작 시 1회 생성. None이면 "키 미설정" -> 1단계는 폴백, 2단계는 건너뜀.
        self._client = client if client is not None else build_llm_client(config)

    @property
    def llm_enabled(self) -> bool:
        return self._client is not None

    def analyze(self, masked_text: str, locale: str) -> AnalyzeResponse:
        text = masked_text.strip()
        if not text:
            return _empty_response()

        base = self._classify(text, locale)               # 1단계
        details = self._extract_details(text, locale, base)  # 2단계 (없으면 None)
        return _to_response(base, details)

    # 1단계: 분류 + 기본 설명
    def _classify(self, text: str, locale: str) -> LlmAnalysis:
        if self._client is None:
            return fallback_analysis(text)

        data = self._generate_json(build_classify_prompt(text, locale, _today()))
        if data is None:
            return fallback_analysis(text)
        try:
            return LlmAnalysis.model_validate(data)
        except ValidationError:
            return fallback_analysis(text)

    # 2단계: 분류된 카테고리에 맞는 상세 정보를 추출한다.
    # 맛집은 상세 추출이 실패해도 Android가 카드로 저장할 수 있어야 하므로
    # 최소 details를 만들고 카카오 Local API 보강까지 시도한다.
    def _extract_details(self, text: str, locale: str, base: LlmAnalysis) -> dict | None:
        stage2 = CATEGORY_STAGE2.get(base.category)
        if stage2 is None or self._client is None:
            # LLM을 사용할 수 없거나 2단계 추출기가 없는 경우에도 맛집은 최소 카드 데이터를 만든다.
            if base.category == "restaurant":
                return self._enrich_details("restaurant", _fallback_restaurant_details(base))
            return None

        prompt = stage2.build_prompt(text, base, locale, _today())
        data = self._generate_json(prompt)
        if data is None:
            # 2단계 LLM 호출 또는 JSON 파싱 실패. 맛집만 최소 details로 대체한다.
            if base.category == "restaurant":
                return self._enrich_details("restaurant", _fallback_restaurant_details(base))
            return None
        try:
            details = stage2.details_model.model_validate(data).model_dump()
        except ValidationError:
            logger.warning("stage-2 details validation failed for category=%s", base.category)
            # 모델 검증 실패는 상세 정보 부족으로 간주한다. 맛집은 검색 가능한 최소 구조로 대체한다.
            if base.category == "restaurant":
                return self._enrich_details("restaurant", _fallback_restaurant_details(base))
            return None
        return self._enrich_details(base.category, details)

    def _enrich_details(self, category: str, details: dict) -> dict:
        if category != "restaurant" or not self._config.kakao_rest_api_key:
            return details

        client = KakaoLocalClient(
            rest_api_key=self._config.kakao_rest_api_key,
            timeout_seconds=self._config.kakao_timeout_seconds,
        )
        return enrich_restaurant_details_with_kakao(details, client)

    def _generate_json(self, prompt: str) -> dict | None:
        assert self._client is not None
        attempts = self._config.llm_max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                raw = self._client.generate(prompt)
                return _parse_json(raw)
            except (LlmError, ValueError) as exc:
                # 오류의 '종류'만 로깅한다. 프롬프트나 모델 출력은 절대 남기지 않는다.
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt,
                    attempts,
                    type(exc).__name__,
                )
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))  # 1초, 2초, 최대 4초로 제한

        return None


def _today() -> str:
    return date.today().isoformat()


def _parse_json(raw: str) -> dict:
    """관용적 JSON 추출: 일부 모델이 여전히 붙이는 ```json 코드펜스를 제거한다."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        first, _, rest = cleaned.partition("\n")
        if first.strip().lower() in {"json", ""}:
            cleaned = rest

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON from LLM: {type(exc).__name__}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON was not an object")
    return parsed


def _to_response(base: LlmAnalysis, details: dict | None) -> AnalyzeResponse:
    # category는 허용 목록(화이트리스트)으로 강제한다. LLM이 엉뚱한 값을 줘도 안전하다.
    category = base.category if base.category in ALLOWED_CATEGORIES else UNKNOWN_CATEGORY

    # 추천 액션은 2단계(카테고리 맞춤)가 있으면 그것을 우선하고, 없으면 1단계 기본값을 쓴다.
    # details 안에 중복 저장하지 않도록 꺼내서 최상위 필드로만 노출한다.
    recommended = base.recommendedAction
    if details is not None:
        stage2_action = details.pop("recommendedAction", None)
        if stage2_action:
            recommended = stage2_action

    return AnalyzeResponse(
        serverMemoId=None,
        title=(base.title[:40] or "New capture"),
        summary=(base.summary or "No content to summarize."),
        category=category,
        recommendedAction=recommended,
        reminderAt=_iso_to_epoch_ms(base.reminderAtIso),
        details=details,
    )

# 로컬 테스트용 fallback 장소명. 2단계 details 실패 시에도 카카오 Local 보강 흐름을 확인하기 위해 고정값을 사용한다.
# 실제 배포 전에는 base.title 기반 후보 추출 또는 LLM details 결과 사용으로 되돌려야 한다.
def _fallback_restaurant_details(base: LlmAnalysis) -> dict:
    # Android 맛집 카드 저장과 카카오 Local 검색에 필요한 최소 details 구조.
    # 1단계 title을 장소명 후보로 사용하되, 검색 정확도가 낮을 수 있어 needsUserReview를 유지한다.
    return {
        "restaurant": {
            "name": "백소정",
            "address": None,
            "roadAddress": None,
            "neighborhood": None,
            "latitude": None,
            "longitude": None,
            "mapProvider": None,
            "mapProviderPlaceId": None,
            "menus": [],
            "estimatedPricePerPersonMin": None,
            "estimatedPricePerPersonMax": None,
            "tags": [],
            "features": [],
            "recommendedActions": [],
        },
        "group": None,
        "confidence": 0.3,
        "needsUserReview": True,
    }

def _iso_to_epoch_ms(value: str | None) -> int | None:
    # LLM은 ISO 문자열만 주고, 실제 epoch ms 변환은 여기서 안전하게 처리한다.
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp() * 1000)
    except (ValueError, OverflowError):
        return None


def _empty_response() -> AnalyzeResponse:
    return AnalyzeResponse(
        title="New capture",
        summary="No content to summarize.",
        category=UNKNOWN_CATEGORY,
        recommendedAction="메모로 저장",
        reminderAt=None,
        details=None,
    )
