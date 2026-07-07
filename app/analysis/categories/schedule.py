from typing import Optional

from pydantic import BaseModel

from app.analysis.prompt import LlmAnalysis

# ── 일정/공지(schedule) 담당자가 채워 확장하는 파일 ─────────────────────────
# 목표 액션: 일정 요약 + 외부 캘린더 연동(초안 생성)
# LLM은 캘린더 초안에 필요한 '필드'만 뽑는다. 실제 캘린더 생성은 안드로이드 몫.


class ScheduleDetails(BaseModel):
    """캘린더 초안 생성에 필요한 필드. (담당자가 자유롭게 추가/수정)"""

    eventTitle: Optional[str] = None   # 캘린더 일정 제목
    displayTime: Optional[str] = None  # 사람이 읽는 일정 일시
    startAtIso: Optional[str] = None   # 시작 일시 (ISO 8601)
    endAtIso: Optional[str] = None     # 종료 일시 (ISO 8601)
    location: Optional[str] = None     # 장소


def build_schedule_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 1단계에서 'schedule'(일정/공지)로 분류되었습니다.
사용자가 효율적인 메모를 만들고, 이후 캘린더 API로 일정을 추가할 수 있도록
가장 중요한 일정 하나를 캘린더 초안 형태로 추출하세요.

JSON 객체 하나만 출력하세요. 설명·마크다운·코드펜스 없이 순수 JSON만 출력하세요.

오늘 날짜: {today_iso}
사용자 로케일: {locale}

추출 규칙:
- eventTitle은 캘린더 제목으로 바로 쓸 수 있게 40자 이내로 작성하세요.
- displayTime은 사용자가 읽기 좋은 한국어 일정 표현으로 작성하세요.
  예: "2026년 7월 3일 (금) 오후 7:00", "2026년 7월 3일 (금)~7월 13일 (월)"
- startAtIso/endAtIso는 캘린더 API에 전달할 ISO 8601 문자열입니다.
- 기간만 있고 시간이 없으면 시작일 00:00:00, 종료일은 마지막 날 23:59:59로 채우세요.
- 시작만 있고 종료가 없으면 endAtIso는 null로 두세요.
- '내일', '이번 주 금요일' 같은 상대 표현은 오늘 날짜 기준으로 계산하세요.
- 후보 일정이 여러 개면 텍스트의 핵심 목적에 가장 가까운 대표 일정을 선택하세요.
- 장소가 명확하면 location에 넣고, 없으면 null로 두세요.
- 텍스트에 없는 정보는 추측하지 말고 null로 두세요.

JSON 스키마:
{{
  "eventTitle": "일정 제목, 없으면 null",
  "displayTime": "사람이 읽는 일정 일시, 없으면 null",
  "startAtIso": "시작 일시 ISO 8601 문자열, 없으면 null",
  "endAtIso": "종료 일시 ISO 8601 문자열, 없으면 null",
  "location": "장소, 없으면 null"
}}

출력 예시:
{{
  "eventTitle": "부천국제영화제 개막",
  "displayTime": "2026년 7월 3일 (금) 오후 7:00",
  "startAtIso": "2026-07-03T19:00:00",
  "endAtIso": null,
  "location": "부천시청 잔디광장"
}}

텍스트:
\"\"\"
{text}
\"\"\"
"""
