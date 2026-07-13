import json
import logging
import time
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.analysis.batch import BatchAnalyzer
from app.analysis.service import AnalysisService
from app.logging_config import new_request_id, request_id_var, setup_logging
from app.models import AnalyzeBatchItem, AnalyzeBatchResponse

# 다른 모듈이 로깅하기 전에 먼저 로거를 구성한다(레벨/포맷/rid 필터).
setup_logging()
logger = logging.getLogger("capturemate.api")

app = FastAPI(title="CaptureMate AI")

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

        # 최초 1회 모델 로드는 수 초 걸릴 수 있어, 시작/완료를 남겨 '멈춘 게 아님'을 알린다.
        logger.info("OCR 엔진 최초 로드 시작 (모델 다운로드가 있을 수 있음)")
        started = time.perf_counter()
        _ocr_engine = PaddleOcrEngine()
        logger.info("OCR 엔진 로드 완료 (%.0fms)", (time.perf_counter() - started) * 1000)
    return _ocr_engine


def _parse_metadata(metadata: str | None) -> list:
    if not metadata:
        return []
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid metadata JSON") from exc
    return parsed if isinstance(parsed, list) else []


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llmEnabled": analysis_service.llm_enabled}


@app.post("/v1/analyze", response_model=AnalyzeBatchResponse)
async def analyze(
    images: Annotated[list[UploadFile], File()],
    locale: Annotated[str, Form()] = "ko-KR",
    metadata: Annotated[str | None, Form()] = None,
) -> AnalyzeBatchResponse:
    """최종 프로덕션 엔드포인트.

    여러 이미지(multipart) -> 백엔드 OCR -> 마스킹 -> 유사 그룹핑 -> LLM -> 그룹별 메모.
    이미지는 메모리에서만 처리하고 저장하지 않는다(즉시 폐기).

    metadata(선택): 이미지 순서와 1:1로 대응하는 JSON 배열.
      예) [{"clientId": "a.png", "capturedAt": 1760000000000}, ...]
      capturedAt이 있으면 시간 근접 그룹핑에 사용된다.
    """
    # 요청 상관관계 ID 설정 — 이 요청의 모든 단계 로그가 같은 [rid]로 묶인다.
    request_id_var.set(new_request_id())
    request_started = time.perf_counter()
    logger.info("analyze 요청 수신: images=%d locale=%s", len(images), locale)

    meta = _parse_metadata(metadata)

    try:
        engine = _get_ocr_engine()
    except ImportError as exc:
        logger.error("OCR 엔진 미설치로 요청 거부(503)")
        raise HTTPException(
            status_code=503,
            detail="OCR engine not installed (pip install -r requirements-ocr.txt)",
        ) from exc

    items: list[AnalyzeBatchItem] = []
    skipped_short = 0
    for index, image in enumerate(images):
        content = await image.read()
        if len(content) > MAX_IMAGE_BYTES:
            logger.warning("이미지 용량 초과(413) idx=%d bytes=%d", index, len(content))
            raise HTTPException(status_code=413, detail=f"image too large: {image.filename}")

        # 이미지별 OCR 실패 격리: 한 장이 깨져도 전체 배치를 실패시키지 않는다.
        try:
            ocr_started = time.perf_counter()
            raw_text = await run_in_threadpool(engine.extract_text, content)
        except Exception as exc:  # OCR 디코드/인식 오류 등
            logger.warning("OCR 실패로 건너뜀 idx=%d (%s): %s", index, image.filename, type(exc).__name__)
            continue
        finally:
            del content  # 이미지 즉시 폐기

        ocr_ms = (time.perf_counter() - ocr_started) * 1000
        # 유용성 안전망: 글자가 거의 없으면(밈/사진) 분석 대상에서 제외.
        if len(raw_text.strip()) < MIN_USEFUL_CHARS:
            skipped_short += 1
            logger.debug("OCR idx=%d 글자부족으로 제외 chars=%d (%.0fms)", index, len(raw_text.strip()), ocr_ms)
            continue

        logger.debug("OCR idx=%d 완료 chars=%d (%.0fms)", index, len(raw_text.strip()), ocr_ms)
        entry = meta[index] if index < len(meta) and isinstance(meta[index], dict) else {}
        client_id = entry.get("clientId") or image.filename or f"image_{index}"
        items.append(
            AnalyzeBatchItem(
                clientId=client_id,
                maskedText=raw_text,
                capturedAt=entry.get("capturedAt"),
            )
        )

    logger.info(
        "OCR 단계 완료: 분석대상 %d/%d장 (글자부족 제외 %d, OCR실패 제외 %d)",
        len(items),
        len(images),
        skipped_short,
        len(images) - len(items) - skipped_short,
    )

    # 마스킹 -> 임베딩 그룹핑 -> 그룹별 LLM. 블로킹 작업이라 스레드풀에서 실행.
    groups = await run_in_threadpool(batch_analyzer.analyze_batch, items, locale)
    logger.info(
        "analyze 요청 완료: groups=%d (전체 %.0fms)",
        len(groups),
        (time.perf_counter() - request_started) * 1000,
    )
    return AnalyzeBatchResponse(groups=groups)
