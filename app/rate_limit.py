import threading
import time

from app.config import settings


class RateLimiter:
    """호출 간 최소 간격을 강제하는 단순 레이트 리미터(스레드 안전).

    무료 티어의 분당 요청 한도(RPM)를 넘겨 429가 나는 것을 막기 위해, 각 API 호출
    직전에 acquire()를 불러 이전 호출과의 간격을 벌린다. min_interval<=0이면 무동작.
    배치는 스레드풀에서 돌기 때문에 여기서의 sleep은 이벤트 루프를 막지 않는다.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min = min_interval_seconds
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        if self._min <= 0:
            return
        with self._lock:
            wait = self._min - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# 임베딩·생성 모든 Gemini 호출에 공유 적용(집계 호출률을 제어).
gemini_rate_limiter = RateLimiter(settings.llm_min_interval_seconds)
