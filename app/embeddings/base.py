from typing import Protocol, runtime_checkable


class EmbeddingError(Exception):
    """임베딩 공급자 호출이 실패했을 때 발생. 메시지에 텍스트 내용은 담지 않는다."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """텍스트를 의미 벡터로 바꾸는 단일 접점. (LlmClient와 같은 패턴)

    새 공급자는 `embed` 하나만 맞추면 되고, factory에서만 교체한다.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 한 번에 임베딩해 각 텍스트의 벡터 리스트를 반환한다.

        입력 순서와 출력 순서는 1:1로 대응한다. 실패 시 EmbeddingError.
        """
        ...
