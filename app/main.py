from fastapi import FastAPI

from app.analysis.batch import BatchAnalyzer
from app.analysis.service import AnalysisService
from app.models import (
    AnalyzeBatchRequest,
    AnalyzeBatchResponse,
    AnalyzeRequest,
    AnalyzeResponse,
)
from app.privacy import mask_text

app = FastAPI(title="CaptureMate AI")

# 시작 시 1회 생성 — LLM 클라이언트 생성/모델 로드가 한 번만 일어나게 한다.
analysis_service = AnalysisService()
# 배치 분석기는 위 분석 서비스를 공유(같은 LLM 클라이언트)하고 임베딩 클라이언트를 추가로 갖는다.
batch_analyzer = BatchAnalyzer(analysis_service=analysis_service)


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


@app.post("/v1/analyze-batch", response_model=AnalyzeBatchResponse)
def analyze_batch(request: AnalyzeBatchRequest) -> AnalyzeBatchResponse:
    """하루치 스크린샷을 한 번에 받아 유사한 것끼리 묶고, 그룹당 메모 1개를 반환한다.

    이미지가 아니라 마스킹된 텍스트 배열을 받는다(온디바이스 OCR 파이프라인용).
    """
    groups = batch_analyzer.analyze_batch(request.items, request.locale)
    return AnalyzeBatchResponse(groups=groups)
