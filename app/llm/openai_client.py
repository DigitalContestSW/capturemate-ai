import logging
import time

from app.llm.base import LlmError
from app.rate_limit import llm_rate_limiter

logger = logging.getLogger("capturemate.llm")


class OpenAiLlmClient:
    """`LlmClient`의 OpenAI 구현체.

    `openai` SDK를 지연 import 하므로, 의존성 미설치/키 없음이어도 앱은 정상 기동해
    키워드 폴백으로 동작할 수 있다(GeminiLlmClient와 동일한 계약).

    JSON 모드(response_format=json_object)로 유효한 JSON 출력을 강제한다 —
    구조화 결과의 1차 방어선. (프롬프트에 'JSON'이라는 단어가 있어야 활성화된다.)
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int = 8192,
    ) -> None:
        from openai import OpenAI

        # max_retries=0: SDK 내부 재시도를 끄고, 재시도/백오프는 상위(_generate_json)로 일원화.
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self._model = model
        self._max_output_tokens = max_output_tokens

    def generate(self, prompt: str) -> str:
        # 내용은 남기지 않고 길이만 — 어느 호출이 느린지/큰지 파악용.
        logger.debug("OpenAI generate 호출: model=%s prompt_chars=%d", self._model, len(prompt))
        llm_rate_limiter.acquire()  # 레이트리밋 완화 (대기가 여기 포함될 수 있음)
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You output only a single valid JSON object."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                # 출력 상한 — 응답이 조용히 잘려 JSON 파싱이 깨지는 것을 통제한다.
                max_completion_tokens=self._max_output_tokens,
                # 유효한 JSON 출력을 강제(프롬프트에 'JSON' 단어 포함 시 활성화).
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # 네트워크 / 공급자 / SDK 오류
            # 오류 종류 + HTTP 상태코드만 남긴다(429=레이트, 500/503=서버 등). 내용은 미기록.
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            detail = type(exc).__name__ + (f"({status})" if status else "")
            logger.warning(
                "OpenAI generate 실패: %s (%.0fms)",
                detail,
                (time.perf_counter() - started) * 1000,
            )
            raise LlmError(f"OpenAI request failed: {detail}") from exc

        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LlmError("OpenAI returned an empty response")
        logger.debug(
            "OpenAI generate 응답: chars=%d (%.0fms)", len(text), (time.perf_counter() - started) * 1000
        )
        return text
