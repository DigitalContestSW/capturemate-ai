import logging
import time

from app.llm.base import LlmError
from app.rate_limit import gemini_rate_limiter

logger = logging.getLogger("capturemate.llm")


class GeminiLlmClient:
    """`LlmClient`의 Google Gemini 구현체.

    `google-genai` SDK를 사용한다. SDK는 지연 import(lazy import)하므로,
    의존성이 설치되지 않았거나 키가 없을 때도 앱이 정상 기동하여
    키워드 폴백으로 동작할 수 있다.
    """

    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._timeout_ms = int(timeout_seconds * 1000)

    def generate(self, prompt: str) -> str:
        from google.genai import types

        # 내용은 남기지 않고 길이만 — 어느 호출이 느린지/큰지 파악용.
        logger.debug("Gemini generate 호출: model=%s prompt_chars=%d", self._model, len(prompt))
        gemini_rate_limiter.acquire()  # 레이트리밋 완화 (대기가 여기 포함될 수 있음)
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    # Gemini에서 유효한 JSON 출력을 강제한다 — 구조화 결과의 1차 방어선.
                    response_mime_type="application/json",
                    temperature=0.2,
                    http_options=types.HttpOptions(timeout=self._timeout_ms),
                ),
            )
        except Exception as exc:  # 네트워크 / 공급자 / SDK 오류
            # 오류 '종류 + HTTP 상태코드'까지 남긴다(429=쿼터/레이트, 503=서버, 500 등).
            # 상태코드·예외 클래스명은 사용자 텍스트를 담지 않아 안전 — 원인 특정에 필수.
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            detail = type(exc).__name__ + (f"({status})" if status else "")
            logger.warning(
                "Gemini generate 실패: %s (%.0fms)",
                detail,
                (time.perf_counter() - started) * 1000,
            )
            raise LlmError(f"Gemini request failed: {detail}") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise LlmError("Gemini returned an empty response")
        logger.debug(
            "Gemini generate 응답: chars=%d (%.0fms)", len(text), (time.perf_counter() - started) * 1000
        )
        return text
