from typing import Protocol, runtime_checkable


class LlmError(Exception):
    """LLM 공급자 호출이 실패하거나 사용할 수 없는 출력을 반환할 때 발생한다.

    메시지에는 오류의 '종류'만 담고 프롬프트/응답 내용은 절대 담지 않는다.
    예외를 통해 PII가 로그로 새어 나가는 것을 막기 위함이다.
    """


@runtime_checkable
class LlmClient(Protocol):
    """모든 LLM 공급자가 구현해야 하는 단일 접점.

    Protocol(구조적 타이핑)을 쓰므로, 새 공급자는 `generate` 메서드만 맞춘
    클래스 하나면 된다 — 상속할 부모 클래스도, factory 외에 수정할 레지스트리도 없다.
    '텍스트 입력 / 텍스트 출력'은 모든 공급자가 지원하는 최소 공통분모이며,
    JSON 강제는 각 클라이언트의 구현 세부사항으로 숨긴다.
    """

    def generate(self, prompt: str) -> str:
        """주어진 `prompt`에 대한 모델의 원시 텍스트 응답을 반환한다.

        구현체는 공급자가 지원하면 JSON 출력을 요청해야 하며,
        실패·타임아웃·빈 출력 시 `LlmError`를 발생시킨다.
        """
        ...
