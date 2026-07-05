from app.config import Settings
from app.embeddings.base import EmbeddingClient
from app.embeddings.gemini_embedding import GeminiEmbeddingClient


def build_embedding_client(config: Settings) -> EmbeddingClient | None:
    """설정에 맞는 임베딩 클라이언트를 생성하거나, 사용할 수 없으면 None.

    None(키 없음/미지원 공급자)이면 그룹핑을 건너뛰고 각 항목을 개별 처리한다.
    """
    if not config.llm_api_key:
        return None

    provider = config.llm_provider.lower()
    if provider == "gemini":
        return GeminiEmbeddingClient(
            api_key=config.llm_api_key,
            model=config.llm_embedding_model,
        )

    # 임베딩 미구현 공급자 -> None (그룹핑 없이 개별 처리)
    return None
