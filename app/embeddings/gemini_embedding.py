from app.embeddings.base import EmbeddingError


class GeminiEmbeddingClient:
    """Google Gemini 임베딩 구현체. SDK는 지연 import.

    텍스트를 '한 개씩' 임베딩한다. 리스트를 통째로 넘기면 SDK가 하나의 콘텐츠로
    묶어 벡터를 1개만 반환하는 문제가 있어, 입력 수 = 벡터 수를 보장하려고 개별 호출한다.
    (하루치 수십 개 규모라 개별 호출 비용/시간은 미미하다.)
    """

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for text in texts:
            try:
                response = self._client.models.embed_content(model=self._model, contents=text)
            except Exception as exc:  # 네트워크 / 공급자 / SDK 오류
                raise EmbeddingError(f"Gemini embedding failed: {type(exc).__name__}") from exc

            embeddings = getattr(response, "embeddings", None)
            if not embeddings:
                raise EmbeddingError("Gemini returned no embedding")
            vectors.append(list(embeddings[0].values))

        return vectors
