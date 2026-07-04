from typing import Optional

from pydantic import BaseModel

from app.analysis.prompt import LlmAnalysis

# ── 학습(study) 담당자가 채워 확장하는 파일 ──────────────────────────────────
# 목표 액션: 스크린샷 내용 요약 + 체크리스트 생성 + 리마인더
# LLM은 체크리스트 항목과 알림 시각을 뽑는다. 알림 예약은 안드로이드(WorkManager) 몫.


class StudyDetails(BaseModel):
    """복습 체크리스트/리마인더에 필요한 필드. (담당자가 자유롭게 추가/수정)"""

    checklist: list[str] = []          # 복습/할 일 항목
    reminderAtIso: Optional[str] = None  # 복습 알림 시각 (ISO 8601)


def build_study_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 '학습'으로 분류되었습니다. 복습 체크리스트와 리마인더를 위해
아래 필드를 추출하세요. JSON 객체 하나만 출력(설명·코드펜스 없이).

오늘 날짜: {today_iso}
'시험 3일 전' 같은 상대 표현은 오늘 날짜 기준으로 계산하세요.

JSON 스키마:
{{
  "checklist": ["복습할 항목1", "복습할 항목2"],
  "reminderAtIso": "복습 알림 시각 ISO 8601 문자열, 없으면 null"
}}

텍스트:
\"\"\"
{text}
\"\"\"
"""
