import re

# 상태바 잔재: 시간(3:53), 배터리(80/80%), 통신(5G/LTE/Wi-Fi)
_STATUS_BAR = re.compile(r"\d{1,2}:\d{2}|\d{1,3}%?|5G|LTE|Wi-?Fi", re.IGNORECASE)
# 기호만 있는 줄(·, ●, —, ... 등)
_SYMBOLS_ONLY = re.compile(r"[\W_]+")


def preprocess_for_embedding(text: str) -> str:
    """임베딩(그룹핑)용으로 텍스트를 정리한다. 노이즈를 걷어 '내용'에 집중시킨다.

    ⚠️ 이 결과는 '그룹핑 판단용'으로만 쓰고, LLM 분석에는 원본 마스킹 텍스트를 그대로
    사용한다(전처리로 마감일·금액 같은 내용이 유실되지 않도록). 그래서 규칙은 보수적이다.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())  # 공백 정규화
        if not line:
            continue
        if _is_noise(line):
            continue
        if line in seen:  # 중복 줄 제거(스크롤 캡처 등)
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines)


def _is_noise(line: str) -> bool:
    # 진짜 내용을 지우지 않도록, 상태바 패턴과 '기호만 있는 줄'만 제거한다.
    if _STATUS_BAR.fullmatch(line):
        return True
    if _SYMBOLS_ONLY.fullmatch(line):
        return True
    return False
