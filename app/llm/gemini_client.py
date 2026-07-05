from app.llm.base import LlmError


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
            # 타입명만 남긴다 — 프롬프트나 응답 내용은 절대 노출하지 않는다.
            raise LlmError(f"Gemini request failed: {type(exc).__name__}") from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise LlmError("Gemini returned an empty response")
        return text
