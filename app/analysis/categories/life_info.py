from typing import Optional

from pydantic import BaseModel

from app.analysis.prompt import LlmAnalysis

# ── 생활정보(life_info) 담당자가 채워 확장하는 파일 ─────────────────────────
# 와이어프레임: 핵심 정보 + 리마인드
#   - 핵심 정보 -> LLM이 keyInfo로 추출 (여기)
#   - 리마인드 시각 -> 1단계 결과의 reminderAt 사용, 알림 예약은 안드로이드


class LifeInfoDetails(BaseModel):
    """생활정보 세부: 핵심 정보. (담당자가 자유롭게 추가/수정)"""

    keyInfo: list[str] = []                     # 놓치면 안 되는 핵심 정보 항목
    recommendedAction: Optional[str] = None     # 카테고리 맞춤 추천 다음 행동


def build_life_info_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 '생활정보'로 분류되었습니다. 사용자가 나중에 참고할
'핵심 정보'를 뽑아 아래 필드를 추출하세요. JSON 객체 하나만 출력(설명·코드펜스 없이).

JSON 스키마:
{{
  "keyInfo": ["핵심 정보 항목1", "핵심 정보 항목2"],
  "recommendedAction": "추천 다음 행동 (예: 정보 저장, 기한 전 알림 설정)"
}}

keyInfo는 금액·조건·기간·대상·방법 등 실제로 도움이 되는 정보를 짧은 문장으로 3~7개 정리하세요.

텍스트:
\"\"\"
{text}
\"\"\"
"""
