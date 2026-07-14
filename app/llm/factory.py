from app.config import Settings
from app.llm.base import LlmClient
from app.llm.gemini_client import GeminiLlmClient


def build_llm_client(config: Settings) -> LlmClient | None:
    """설정에 맞는 LLM 클라이언트를 생성하거나, LLM을 쓸 수 없으면 None을 반환한다.

    None(키 없음)은 오류가 아니다. 분석 서비스가 이를 "키워드 폴백을 써라"로
    해석하므로, 키가 준비되기 전에도 API가 정상 동작한다.

    나중에 공급자를 추가하는 것은 이곳에 두 줄 + 새 클라이언트 클래스 하나면 된다:
        if provider == "claude":
            return ClaudeLlmClient(...)
    """
    if not config.llm_api_key:
        return None

    provider = config.llm_provider.lower()
    if provider == "gemini":
        return GeminiLlmClient(
            api_key=config.llm_api_key,
            model=config.llm_model,
            timeout_seconds=config.llm_timeout_seconds,
        )
    if provider == "openai":
        # 지연 import — openai SDK 미설치 환경(gemini 전용 배포)에서도 이 모듈이 로드되게 한다.
        from app.llm.openai_client import OpenAiLlmClient

        return OpenAiLlmClient(
            api_key=config.llm_api_key,
            model=config.llm_model,
            timeout_seconds=config.llm_timeout_seconds,
        )

    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
