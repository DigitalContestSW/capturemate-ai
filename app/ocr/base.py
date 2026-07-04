from typing import Protocol, runtime_checkable


class OcrError(Exception):
    """OCR 엔진이 이미지를 디코드/인식하지 못할 때 발생."""


@runtime_checkable
class OcrEngine(Protocol):
    """모든 백엔드 OCR 엔진이 구현해야 하는 단일 접점.

    LLM의 `LlmClient`와 같은 패턴. 엔진(PaddleOCR/EasyOCR 등)을 이 인터페이스
    뒤로 숨기면 비교 실험에서 구현체만 바꿔 끼울 수 있다.

    입력은 원시 이미지 bytes(디스크/S3 저장 없이 메모리에서 처리),
    출력은 추출된 전체 텍스트.
    """

    def extract_text(self, image_bytes: bytes) -> str:
        ...
