import logging
import os
import sys
import uuid
from contextvars import ContextVar

# 요청별 상관관계 ID. 엔드포인트에서 set하면 같은 요청의 모든 로그에 자동으로 붙는다.
# contextvars는 run_in_threadpool(스레드풀)로도 복사되므로, 배치/LLM 단계 로그까지
# 같은 rid로 묶여 "이 요청이 어디까지 갔는지"를 한 줄 grep으로 따라갈 수 있다.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_CONFIGURED = False


class _RequestIdFilter(logging.Filter):
    """모든 로그 레코드에 현재 요청의 rid를 주입한다(포맷에서 %(request_id)s로 사용)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    """capturemate.* 로거를 한 번만 구성한다.

    - 레벨은 LOG_LEVEL 환경변수(기본 INFO). 자세히 보려면 LOG_LEVEL=DEBUG.
    - uvicorn root 핸들러로 중복 출력되지 않게 propagate=False로 자체 핸들러만 사용.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger("capturemate")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def new_request_id() -> str:
    """짧은 8자리 요청 ID."""
    return uuid.uuid4().hex[:8]
