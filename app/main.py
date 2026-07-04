from fastapi import FastAPI

from app.analysis.service import AnalysisService
from app.models import AnalyzeRequest, AnalyzeResponse
from app.privacy import mask_text

app = FastAPI(title="CaptureMate AI")

# 시작 시 1회 생성 — LLM 클라이언트 생성/모델 로드가 한 번만 일어나게 한다.
analysis_service = AnalysisService()


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
