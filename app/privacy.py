import logging
import re
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger("capturemate.privacy")

# 절대 복원하지 않는 하드 PII (고정 placeholder, 매핑에 담지 않음 -> 영원히 가려짐).
_HARD_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("RRN", re.compile(r"\b\d{6}[-\s]?[1-4]\d{6}\b")),        # 주민등록번호
    ("CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),     # 카드번호 4-4-4-4
]

# 문맥에 따라 복원될 수 있는 소프트 PII (인덱스 placeholder + 원본 매핑 보관).
_SOFT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    (
        "PHONE",
        re.compile(
            r"\b01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}\b"   # 휴대폰
            r"|\b0\d{1,2}[-\s.]?\d{3,4}[-\s.]?\d{4}\b"    # 지역번호(회사 등)
        ),
    ),
]

# Presidio(NER) 탐지기는 무겁고 선택적이라 최초 사용 시 지연 로드.
_ner_detector = None


@dataclass
class MaskResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # "[PHONE_1]" -> "02-123-4567"


def mask_text(text: str, namespace: str = "") -> MaskResult:
    """민감정보를 마스킹하고, 복원 가능한(소프트) 항목은 placeholder->원본 매핑도 반환한다.

    - (선택) NER: use_presidio=true면 이름/주소 등 자유형 PII를 먼저 소프트 토큰으로.
    - 하드(주민번호/카드): 고정 placeholder, 매핑 없음 -> 절대 복원 안 됨.
    - 소프트(이메일/전화/이름/주소): [PHONE_1] 인덱스 placeholder + 매핑.
      LLM이 문맥상 필요해 결과에 그 토큰을 포함하면 나중에 원본으로 복원한다.
    - namespace: 배치에서 항목 간 토큰 충돌 방지 접두어(예: "0_").
    """
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    masked = text

    # 0) (선택) NER 마스킹 — 실패해도 정규식만으로 계속 진행(무중단).
    if settings.use_presidio:
        try:
            masked = _mask_ner(masked, namespace, mapping, counters)
        except Exception as exc:  # 미설치/모델없음/런타임 오류 등
            logger.warning("NER 마스킹 생략(정규식만 적용): %s", type(exc).__name__)

    # 1) 하드 정규식
    for label, pattern in _HARD_RULES:
        masked = pattern.sub(f"[{label}]", masked)

    # 2) 소프트 정규식
    for label, pattern in _SOFT_RULES:
        masked = _sub_soft(masked, label, pattern, namespace, mapping, counters)

    return MaskResult(text=masked.strip(), mapping=mapping)


def restore_text(text: str, mapping: dict[str, str]) -> str:
    """text에 등장하는 placeholder를 원본 값으로 되돌린다(매핑에 있는 소프트 토큰만)."""
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


def _make_token(label: str, namespace: str, counters: dict[str, int]) -> str:
    counters[label] = counters.get(label, 0) + 1
    return f"[{label}_{namespace}{counters[label]}]"


def _sub_soft(
    text: str,
    label: str,
    pattern: re.Pattern[str],
    namespace: str,
    mapping: dict[str, str],
    counters: dict[str, int],
) -> str:
    def repl(match: re.Match) -> str:
        token = _make_token(label, namespace, counters)
        mapping[token] = match.group(0)
        return token

    return pattern.sub(repl, text)


def _mask_ner(
    text: str,
    namespace: str,
    mapping: dict[str, str],
    counters: dict[str, int],
) -> str:
    spans = _get_ner_detector().detect(text)  # [(start, end, label)]
    # 오른쪽(뒤)부터 치환해야 앞쪽 인덱스가 밀리지 않는다.
    for start, end, label in sorted(spans, key=lambda s: s[0], reverse=True):
        token = _make_token(label, namespace, counters)
        mapping[token] = text[start:end]
        text = text[:start] + token + text[end:]
    return text


def _get_ner_detector():
    global _ner_detector
    if _ner_detector is None:
        from app.privacy_ner import PresidioNerDetector

        _ner_detector = PresidioNerDetector()
    return _ner_detector
