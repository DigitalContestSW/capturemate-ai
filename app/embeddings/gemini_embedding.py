import logging
import time

from app.embeddings.base import EmbeddingError
from app.rate_limit import gemini_rate_limiter

logger = logging.getLogger("capturemate.embeddings")


class GeminiEmbeddingClient:
    """Google Gemini 임베딩 구현체. SDK는 지연 import.

    텍스트를 '한 개씩' 임베딩한다. 리스트를 통째로 넘기면 SDK가 하나의 콘텐츠로
    묶어 벡터를 1개만 반환하는 문제가 있어, 입력 수 = 벡터 수를 보장하려고 개별 호출한다.
    (하루치 수십 개 규모라 개별 호출 비용/시간은 미미하다.)
    """

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 15.0) -> None:
        from google import genai
        from google.genai import types

        # 호출 1건당 타임아웃을 Client에 걸어, provider 불안정 시 무한정 매달리는 것을 막는다.
        # (타임아웃이 없으면 SDK 내부 재시도/대기로 한 요청이 수십 분 hang될 수 있다.)
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        n = len(texts)
        logger.debug("Gemini 임베딩: model=%s 개별 호출 %d건 시작", self._model, n)
        started = time.perf_counter()
        vectors: list[list[float]] = []
        for i, text in enumerate(texts):
            gemini_rate_limiter.acquire()  # 레이트리밋 완화
            try:
                response = self._client.models.embed_content(model=self._model, contents=text)
            except Exception as exc:  # 네트워크 / 공급자 / SDK 오류
                # 몇 번째에서 실패했는지 남겨 어디까지 진행됐는지 파악 (내용은 미기록).
                logger.debug("Gemini 임베딩 실패: %d/%d번째 (%s)", i + 1, n, type(exc).__name__)
                raise EmbeddingError(f"Gemini embedding failed: {type(exc).__name__}") from exc

            embeddings = getattr(response, "embeddings", None)
            if not embeddings:
                logger.debug("Gemini 임베딩 빈 응답: %d/%d번째", i + 1, n)
                raise EmbeddingError("Gemini returned no embedding")
            vectors.append(list(embeddings[0].values))

        logger.debug(
            "Gemini 임베딩 완료: %d벡터 (%.0fms)", len(vectors), (time.perf_counter() - started) * 1000
        )
        return vectors
