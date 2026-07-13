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
from app.privacy import restore_text

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
            logger.debug("빈 텍스트 -> empty 응답(분석 건너뜀)")
            return _empty_response()

        base = self._classify(text, locale)               # 1단계 (분류 + 유용성)
        # 유용하지 않으면 2단계(세부 추출)를 건너뛴다 — 잡담/밈 등에 LLM 낭비 방지.
        if base.isUseful:
            details = self._extract_details(text, locale, base)
        else:
            logger.debug("isUseful=false -> 2단계(세부추출) 건너뜀")
            details = None
        return _to_response(base, details)

    # ── 1단계: 분류 + 기본 설명 ──────────────────────────────────────────────
    def _classify(self, text: str, locale: str) -> LlmAnalysis:
        if self._client is None:
            logger.info("1단계: LLM 미사용 -> 키워드 폴백 분류")
            return fallback_analysis(text)

        logger.info("1단계(분류) 시작: chars=%d", len(text))
        started = time.perf_counter()
        data = self._generate_json(build_classify_prompt(text, locale, _today()))
        elapsed = (time.perf_counter() - started) * 1000
        if data is None:
            logger.warning("1단계 실패 -> 키워드 폴백 사용 (%.0fms)", elapsed)
            return fallback_analysis(text)
        try:
            result = LlmAnalysis.model_validate(data)
        except ValidationError:
            logger.warning("1단계 검증 실패 -> 키워드 폴백 사용 (%.0fms)", elapsed)
            return fallback_analysis(text)
        logger.info(
            "1단계 완료: category=%s isUseful=%s (%.0fms)", result.category, result.isUseful, elapsed
        )
        return result

    # ── 2단계: 카테고리 전용 세부 추출 (선택) ────────────────────────────────
    def _extract_details(self, text: str, locale: str, base: LlmAnalysis) -> dict | None:
        stage2 = CATEGORY_STAGE2.get(base.category)
        if stage2 is None or self._client is None:
            logger.debug("2단계 없음(카테고리=%s 미등록 또는 LLM 미사용) -> 건너뜀", base.category)
            return None  # 미등록 카테고리이거나 LLM 미사용 -> 2단계 건너뜀

        logger.info("2단계(세부추출) 시작: category=%s", base.category)
        started = time.perf_counter()
        prompt = stage2.build_prompt(text, base, locale, _today())
        data = self._generate_json(prompt)
        elapsed = (time.perf_counter() - started) * 1000
        if data is None:
            logger.warning("2단계 실패(응답 없음) category=%s -> details 생략 (%.0fms)", base.category, elapsed)
            return None
        try:
            details = stage2.details_model.model_validate(data).model_dump()
        except ValidationError:
            logger.warning("2단계 검증 실패 category=%s -> details 생략 (%.0fms)", base.category, elapsed)
            return None
        logger.info("2단계 완료: category=%s (%.0fms)", base.category, elapsed)
        return self._enrich_details(base.category, details)

    # ── 공통: JSON 응답 생성(재시도 포함) ───────────────────────────────────
    def _enrich_details(self, category: str, details: dict) -> dict:
        if category != "restaurant" or not self._config.kakao_rest_api_key:
            return details

        logger.info("카카오 로컬 보강 시작: category=%s", category)
        started = time.perf_counter()
        client = KakaoLocalClient(
            rest_api_key=self._config.kakao_rest_api_key,
            timeout_seconds=self._config.kakao_timeout_seconds,
        )
        enriched = enrich_restaurant_details_with_kakao(details, client)
        logger.info("카카오 로컬 보강 완료 (%.0fms)", (time.perf_counter() - started) * 1000)
        return enriched

    def _generate_json(self, prompt: str) -> dict | None:
        assert self._client is not None
        attempts = self._config.llm_max_retries + 1

        for attempt in range(1, attempts + 1):
            call_started = time.perf_counter()
            try:
                raw = self._client.generate(prompt)
                parsed = _parse_json(raw)
                logger.debug(
                    "LLM 호출 성공 attempt=%d/%d (%.0fms)",
                    attempt,
                    attempts,
                    (time.perf_counter() - call_started) * 1000,
                )
                return parsed
            except (LlmError, ValueError) as exc:
                # 메시지는 우리가 만든 안전한 문자열(상태코드/예외클래스명)뿐 — 프롬프트/출력 미포함.
                # 그래야 429(쿼터)·503(서버)·JSON파싱실패를 로그에서 바로 구분한다.
                logger.warning(
                    "LLM call attempt %d/%d failed: %s (%.0fms)",
                    attempt,
                    attempts,
                    exc,
                    (time.perf_counter() - call_started) * 1000,
                )
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))  # 1초, 2초, 최대 4초로 제한

        logger.warning("LLM 호출 최종 실패: %d회 시도 모두 실패", attempts)
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
    # category는 허용 목록(화이트리스트)으로 강제 — LLM이 엉뚱한 값을 줘도 안전.
    category = base.category if base.category in ALLOWED_CATEGORIES else UNKNOWN_CATEGORY

    # 어느 서비스 카테고리에도 속하지 않으면(unknown) 저장할 가치가 없다고 보고 isUseful=false.
    is_useful = base.isUseful and category != UNKNOWN_CATEGORY

    # 추천 액션: 2단계(카테고리 맞춤)가 있으면 그것을 우선, 없으면 1단계 기본값.
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
        isUseful=is_useful,
        recommendedAction=recommended,
        reminderAt=_iso_to_epoch_ms(base.reminderAtIso),
        details=details,
    )


def restore_in_response(response: AnalyzeResponse, mapping: dict[str, str]) -> AnalyzeResponse:
    """응답의 문자열 필드에 남은 placeholder를 원본으로 복원한다.

    LLM이 문맥상 필요하다고 판단해 결과에 포함한 소프트 토큰([PHONE_1] 등)만
    실제 값으로 되돌아간다. 하드 토큰([RRN]/[CARD])은 매핑에 없어 그대로 가려진다.
    """
    if not mapping:
        return response

    details = response.details
    if details:
        details = {key: _restore_value(value, mapping) for key, value in details.items()}

    return response.model_copy(
        update={
            "title": restore_text(response.title, mapping),
            "summary": restore_text(response.summary, mapping),
            "recommendedAction": (
                restore_text(response.recommendedAction, mapping)
                if response.recommendedAction
                else response.recommendedAction
            ),
            "details": details,
        }
    )


def _restore_value(value, mapping: dict[str, str]):
    if isinstance(value, str):
        return restore_text(value, mapping)
    if isinstance(value, list):
        return [restore_text(v, mapping) if isinstance(v, str) else v for v in value]
    return value


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
        isUseful=False,  # 내용이 없으면 저장할 가치도 없음
        recommendedAction="메모로 저장",
        reminderAt=None,
        details=None,
    )
