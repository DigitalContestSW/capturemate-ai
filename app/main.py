import json
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.analysis.batch import BatchAnalyzer
from app.analysis.service import AnalysisService
from app.auth import (
    get_current_user,
    issue_access_token_from_refresh,
    issue_token_pair,
    verify_google_id_token,
)
from app.config import settings
from app.models import (
    AnalyzeBatchItem,
    AnalyzeBatchResponse,
    AuthTokenResponse,
    GoogleAuthRequest,
    RefreshTokenRequest,
)

logger = logging.getLogger("capturemate.api")

app = FastAPI(title="CaptureMate AI")
bearer_scheme = HTTPBearer(auto_error=False)

# 시작 시 1회 생성 — LLM 클라이언트 생성/모델 로드가 한 번만 일어나게 한다.
analysis_service = AnalysisService()
# 배치 분석기는 위 분석 서비스를 공유(같은 LLM 클라이언트)하고 임베딩 클라이언트를 추가로 갖는다.
batch_analyzer = BatchAnalyzer(analysis_service=analysis_service)

# OCR 엔진은 무겁고(모델 로드) 선택적 의존성이라, 최초 요청 때 지연 로드한다.
_ocr_engine = None
MAX_IMAGE_BYTES = 12 * 1024 * 1024
# 서버측 유용성 안전망: OCR 텍스트가 이보다 짧으면 밈/사진으로 보고 분석에서 제외.
MIN_USEFUL_CHARS = 10


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from app.ocr.paddle_engine import PaddleOcrEngine

        _ocr_engine = PaddleOcrEngine()
    return _ocr_engine


def _parse_metadata(metadata: str | None) -> list:
    if not metadata:
        return []
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid metadata JSON") from exc
    return parsed if isinstance(parsed, list) else []


def _require_user(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)]):
    return get_current_user(credentials)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llmEnabled": analysis_service.llm_enabled}


@app.post("/v1/auth/google", response_model=AuthTokenResponse)
async def authenticate_with_google(request: GoogleAuthRequest) -> AuthTokenResponse:
    google_claims = await run_in_threadpool(verify_google_id_token, request.idToken)
    access_token, refresh_token = issue_token_pair(google_claims)
    return AuthTokenResponse(
        accessToken=access_token,
        refreshToken=refresh_token,
        accessExpiresIn=settings.jwt_access_ttl_seconds,
        refreshExpiresIn=settings.jwt_refresh_ttl_seconds,
    )


@app.post("/v1/auth/refresh", response_model=AuthTokenResponse)
def refresh_access_token(request: RefreshTokenRequest) -> AuthTokenResponse:
    access_token = issue_access_token_from_refresh(request.refreshToken)
    return AuthTokenResponse(
        accessToken=access_token,
        accessExpiresIn=settings.jwt_access_ttl_seconds,
    )


@app.post("/v1/analyze", response_model=AnalyzeBatchResponse)
async def analyze(
    images: Annotated[list[UploadFile], File()],
    locale: Annotated[str, Form()] = "ko-KR",
    metadata: Annotated[str | None, Form()] = None,
    _user=Depends(_require_user),
) -> AnalyzeBatchResponse:
    """최종 프로덕션 엔드포인트.

    여러 이미지(multipart) -> 백엔드 OCR -> 마스킹 -> 유사 그룹핑 -> LLM -> 그룹별 메모.
    이미지는 메모리에서만 처리하고 저장하지 않는다(즉시 폐기).

    metadata(선택): 이미지 순서와 1:1로 대응하는 JSON 배열.
      예) [{"clientId": "a.png", "capturedAt": 1760000000000}, ...]
      capturedAt이 있으면 시간 근접 그룹핑에 사용된다.
    """
    meta = _parse_metadata(metadata)

    try:
        engine = _get_ocr_engine()
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="OCR engine not installed (pip install -r requirements-ocr.txt)",
        ) from exc

    items: list[AnalyzeBatchItem] = []
    for index, image in enumerate(images):
        content = await image.read()
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"image too large: {image.filename}")

        # 이미지별 OCR 실패 격리: 한 장이 깨져도 전체 배치를 실패시키지 않는다.
        try:
            raw_text = await run_in_threadpool(engine.extract_text, content)
        except Exception as exc:  # OCR 디코드/인식 오류 등
            logger.warning("OCR 실패로 건너뜀 (%s): %s", image.filename, type(exc).__name__)
            continue
        finally:
            del content  # 이미지 즉시 폐기

        # 유용성 안전망: 글자가 거의 없으면(밈/사진) 분석 대상에서 제외.
        if len(raw_text.strip()) < MIN_USEFUL_CHARS:
            continue

        entry = meta[index] if index < len(meta) and isinstance(meta[index], dict) else {}
        client_id = entry.get("clientId") or image.filename or f"image_{index}"
        items.append(
            AnalyzeBatchItem(
                clientId=client_id,
                maskedText=raw_text,
                capturedAt=entry.get("capturedAt"),
            )
        )

    # 마스킹 -> 임베딩 그룹핑 -> 그룹별 LLM. 블로킹 작업이라 스레드풀에서 실행.
    groups = await run_in_threadpool(batch_analyzer.analyze_batch, items, locale)
    return AnalyzeBatchResponse(groups=groups)
