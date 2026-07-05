from app.ocr.base import OcrError


def _texts_from_result(result) -> list[str]:
    """PaddleOCR 결과에서 텍스트만 추출한다.

    PaddleOCR 3.x(PP-OCRv5)는 결과 객체마다 'rec_texts'(문자열 리스트)를 담는다.
    2.x는 [[ [box, (text, conf)], ... ]] 형태라, 3.x를 먼저 시도하고 비면 2.x로 폴백한다.
    """
    texts: list[str] = []

    # PaddleOCR 3.x: 각 결과의 'rec_texts'
    for res in result or []:
        rec = None
        try:
            rec = res["rec_texts"]
        except (TypeError, KeyError, IndexError):
            rec = None
        if rec is None:
            getter = getattr(res, "get", None)
            if callable(getter):
                rec = getter("rec_texts")
        if isinstance(rec, (list, tuple)):
            texts.extend(t for t in rec if isinstance(t, str))
    if texts:
        return texts

    # PaddleOCR 2.x 폴백: 중첩 구조에서 (text, conf) 쌍만 골라낸다.
    def walk(node) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], (int, float)):
                texts.append(node[0])
                return
            for item in node:
                walk(item)

    walk(result)
    return texts


class PaddleOcrEngine:
    """`OcrEngine`의 PaddleOCR 구현체.

    무거운 모델 로드를 피하려고 SDK는 지연 import 하고, 모델은 인스턴스 1개당
    한 번만 로드한다(최초 생성 시 모델 다운로드가 일어날 수 있음).
    """

    def __init__(self, lang: str = "korean") -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=lang,
            # 검출은 모바일(빠름). 실제 품질 문제는 검출이 아니라 인식모델이었다.
            text_detection_model_name="PP-OCRv5_mobile_det",
            # ★ 핵심: 한국어 인식모델을 '명시적으로' 지정.
            #   검출모델을 지정하면 lang이 인식모델 자동선택에 적용되지 않아,
            #   기본(중국어) 인식모델이 쓰여 한글이 빈칸/한자로 나왔다.
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            # 속도↔정확도 다이얼: 검출 입력 긴 변 상한. 한글 크롭 해상도도 확보하려고
            #   모바일 검출은 가벼우니 2048로 넉넉히 준다. 느리면 낮추면 된다.
            text_det_limit_side_len=2048,
            text_det_limit_type="max",
            cpu_threads=10,
            # 스크린샷엔 불필요한 문서 전처리 단계 비활성 (더 빠르고 단순)
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # PaddlePaddle의 oneDNN CPU 추론 버그(NotImplementedError) 회피
            enable_mkldnn=False,
        )

    def extract_text(self, image_bytes: bytes) -> str:
        import cv2
        import numpy as np

        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise OcrError("이미지를 디코드할 수 없습니다")

        try:
            result = self._ocr.ocr(img)
        except Exception as exc:  # 엔진 내부 오류
            raise OcrError(f"PaddleOCR failed: {type(exc).__name__}") from exc

        # 줄 단위 텍스트를 위→아래 순서 그대로 이어 붙인다.
        return "\n".join(_texts_from_result(result))
