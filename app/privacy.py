import re

# 서버측 2차 마스킹 (심층 방어).
# 안드로이드 클라이언트가 1차로 마스킹하지만, 그게 완벽했다고 신뢰하지 않는다.
# 그래서 텍스트가 LLM에 닿기 직전에 서버에서 한 번 더 마스킹한다.
# placeholder는 클라이언트와 동일하게 맞춰 두 계층의 일관성을 유지한다.
# LLM이 필요로 하는 유용한 정보(날짜, 가격 등)를 지우지 않도록 규칙은 일부러 좁게 잡았다.
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{6}[-\s]?[1-4]\d{6}\b"), "[RRN]"),           # 주민등록번호
    (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CARD]"),        # 카드번호 4-4-4-4
]


def mask_text(text: str) -> str:
    masked = text
    for pattern, placeholder in _RULES:
        masked = pattern.sub(placeholder, masked)
    return masked.strip()
