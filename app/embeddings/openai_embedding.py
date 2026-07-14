import logging
import time

from app.embeddings.base import EmbeddingError
from app.rate_limit import llm_rate_limiter

logger = logging.getLogger("capturemate.embeddings")


class OpenAiEmbeddingClient:
    """OpenAI 임베딩 구현체. SDK는 지연 import.

    OpenAI 임베딩은 리스트 입력을 한 번에 처리하고 '입력 순서대로' 반환하므로
    배치로 1회만 호출한다(Gemini와 달리 개별 호출이 필요 없어 빠르다).
    입력 수 = 벡터 수를 검증해 그룹핑이 항목을 누락하지 않도록 한다.
    """

    def __init__(self, api_key: str, model: str, timeout_seconds: float = 15.0) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        n = len(texts)
        logger.debug("OpenAI 임베딩: model=%s 배치 %d건", self._model, n)
        started = time.perf_counter()
        llm_rate_limiter.acquire()  # 레이트리밋 완화
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except Exception as exc:  # 네트워크 / 공급자 / SDK 오류
            logger.debug("OpenAI 임베딩 실패 (%s)", type(exc).__name__)
            raise EmbeddingError(f"OpenAI embedding failed: {type(exc).__name__}") from exc

        # 응답이 index 순서를 보장하지만, 안전하게 정렬 후 벡터만 추출한다.
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != n:
            raise EmbeddingError("OpenAI returned mismatched embedding count")
        vectors = [list(item.embedding) for item in data]

        logger.debug(
            "OpenAI 임베딩 완료: %d벡터 (%.0fms)", len(vectors), (time.perf_counter() - started) * 1000
        )
        return vectors
