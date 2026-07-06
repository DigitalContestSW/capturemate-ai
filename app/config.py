import os
from dataclasses import dataclass

from dotenv import load_dotenv

# .env 파일이 있으면 환경변수로 읽어들인다(없어도 무시). 실제 OS 환경변수가 우선한다.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """환경변수에서 읽어오는 런타임 설정.

    공급자별 설정값을 모두 코드가 아닌 이곳(환경변수)에 두는 것이,
    비즈니스 로직을 건드리지 않고 LLM 공급자를 교체할 수 있게 하는 핵심이다.
    하드코딩이 없고, API 키는 절대 저장소에 커밋되지 않는다.
    """

    llm_provider: str
    llm_model: str
    llm_embedding_model: str
    llm_api_key: str | None
    llm_timeout_seconds: float
    llm_max_retries: int
    # 그룹핑(유사 스크린샷 묶기) 파라미터
    grouping_threshold: float      # 최종 임계값 (cosine × timeWeight ≥ 이 값이면 같은 그룹)
    grouping_tau_seconds: float    # 시간 가중치 감쇠 특성 시간(초). 클수록 시간 영향 약화
    grouping_w_min: float          # 시간 가중치 하한(0~1). 0이면 완전 감쇠 허용
    kakao_rest_api_key: str | None
    kakao_timeout_seconds: float


def load_settings() -> Settings:
    return Settings(
        # 지금은 "gemini". 나중에 호출부 변경 없이 "claude" / "openai"로 교체 가능.
        llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
        # 계정에서 실제 무료 티어 모델명을 확인한 뒤 사용할 것.
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash"),
        # 임베딩(유사도) 전용 모델. 설치된 SDK/계정에서 사용 가능한 이름인지 확인할 것.
        llm_embedding_model=os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-004"),
        # 공급자에 종속되지 않는 이름. None이면 서비스가 폴백을 사용한다.
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        grouping_threshold=float(os.getenv("GROUPING_THRESHOLD", "0.7")),
        grouping_tau_seconds=float(os.getenv("GROUPING_TAU_SECONDS", "300")),
        grouping_w_min=float(os.getenv("GROUPING_W_MIN", "0.0")),
        kakao_rest_api_key=os.getenv("KAKAO_REST_API_KEY"),
        kakao_timeout_seconds=float(os.getenv("KAKAO_TIMEOUT_SECONDS", "3")),
    )


settings = load_settings()
