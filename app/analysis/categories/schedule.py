from typing import Optional

from pydantic import BaseModel

from app.analysis.prompt import LlmAnalysis

# ── 일정/공지(schedule) 담당자가 채워 확장하는 파일 ─────────────────────────
# 목표 액션: 일정 요약 + 외부 캘린더 연동(초안 생성)
# LLM은 캘린더 초안에 필요한 '필드'만 뽑는다. 실제 캘린더 생성은 안드로이드 몫.


class ScheduleDetails(BaseModel):
    """캘린더 초안 생성에 필요한 필드. (담당자가 자유롭게 추가/수정)"""

    startAtIso: Optional[str] = None   # 시작 일시 (ISO 8601)
    endAtIso: Optional[str] = None     # 종료 일시 (ISO 8601)
    location: Optional[str] = None     # 장소


def build_schedule_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 '일정/공지'로 분류되었습니다. 캘린더 초안 생성을 위해
아래 필드를 추출하세요. JSON 객체 하나만 출력(설명·코드펜스 없이).

오늘 날짜: {today_iso}
'내일', '이번 주 금요일' 같은 상대 표현은 오늘 날짜 기준으로 계산하세요.

JSON 스키마:
{{
  "startAtIso": "시작 일시 ISO 8601 문자열, 없으면 null",
  "endAtIso": "종료 일시 ISO 8601 문자열, 없으면 null",
  "location": "장소, 없으면 null"
}}

텍스트:
\"\"\"
{text}
\"\"\"
"""
