from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.analysis.service import AnalysisService
from app.models import AnalyzeImageResponse, AnalyzeRequest, AnalyzeResponse
from app.privacy import mask_text

app = FastAPI(title="CaptureMate AI")

# 시작 시 1회 생성 — LLM 클라이언트 생성/모델 로드가 한 번만 일어나게 한다.
analysis_service = AnalysisService()

# OCR 엔진은 무겁고(모델 로드) 선택적 의존성이라, 최초 요청 때 지연 로드한다.
# 이렇게 하면 PaddleOCR 미설치여도 나머지 API(/v1/analyze)는 정상 기동한다.
_ocr_engine = None
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from app.ocr.paddle_engine import PaddleOcrEngine

        _ocr_engine = PaddleOcrEngine()
    return _ocr_engine


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llmEnabled": analysis_service.llm_enabled}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    # 2차 마스킹: 클라이언트가 완벽히 마스킹했다고 신뢰하지 않는다.
    masked = mask_text(request.maskedText)
    # 동기 `def` 엔드포인트 -> FastAPI가 스레드풀에서 실행하므로, 블로킹되는
    # LLM 호출이 이벤트 루프(다른 요청)를 막지 않는다.
    return analysis_service.analyze(masked, request.locale)


@app.post("/v1/analyze-image", response_model=AnalyzeImageResponse)
async def analyze_image(
    image: UploadFile = File(...),
    locale: str = Form("ko-KR"),
) -> AnalyzeImageResponse:
    """백엔드 OCR 전체 파이프라인 테스트: 이미지 -> OCR -> 마스킹 -> LLM.

    이미지는 메모리에서만 처리하고 저장하지 않는다(즉시 폐기).
    """
    content = await image.read()
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")

    try:
        engine = _get_ocr_engine()
        # OCR은 CPU 블로킹 작업 -> 스레드풀에서 실행해 이벤트 루프를 막지 않는다.
        raw_text = await run_in_threadpool(engine.extract_text, content)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="OCR engine not installed (pip install -r requirements-ocr.txt)",
        ) from exc
    del content  # 이미지 즉시 폐기

    masked = mask_text(raw_text)               # 서버측 마스킹
    analysis = analysis_service.analyze(masked, locale)  # LLM 2단계 분석
    return AnalyzeImageResponse(ocrText=raw_text, maskedText=masked, analysis=analysis)
