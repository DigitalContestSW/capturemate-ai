from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.analysis.prompt import LlmAnalysis

# ── 일정/공지(schedule) 담당자가 채워 확장하는 파일 ─────────────────────────
# 와이어프레임: 일정 초안 생성(외부 캘린더 연동) + 리마인드 알림
#   - 일정 초안 재료(시작/종료/장소) -> LLM 추출 (여기). 실제 캘린더 생성은 안드로이드.
#   - 리마인드 시각 -> 1단계 결과의 reminderAt 사용.


class ScheduleDeadline(BaseModel):
    """일정/공지 카드에 표시할 마감 정보."""

    label: str = "마감"
    date: Optional[str] = None                   # YYYY-MM-DD
    time: Optional[str] = None                   # HH:mm
    displayText: Optional[str] = None            # 예: "2026년 7월 10일 23:59"


class ScheduleEventDate(BaseModel):
    """일정/공지 카드에 표시할 행사/시작일 정보."""

    date: Optional[str] = None                   # YYYY-MM-DD
    dayOfWeek: Optional[str] = None              # 월/화/수/목/금/토/일
    displayText: Optional[str] = None            # 예: "2026년 7월 10일 (금)"


class ScheduleCalendarAction(BaseModel):
    """클라이언트가 캘린더 추가 버튼을 렌더링할 때 쓰는 표시 정보."""

    label: str = "구글 캘린더에 추가"
    enabled: bool = False


class ScheduleDetails(BaseModel):
    """일정/공지 세부: 카드 렌더링 + 캘린더 초안 생성에 필요한 필드."""

    type: Literal["schedule"] = "schedule"
    title: Optional[str] = None                  # 카드 제목
    source: str = "AI 추출"
    deadline: Optional[ScheduleDeadline] = None
    eventDate: Optional[ScheduleEventDate] = None
    calendarAction: ScheduleCalendarAction = Field(default_factory=ScheduleCalendarAction)
    startAtIso: Optional[str] = None            # 시작 일시 (ISO 8601)
    endAtIso: Optional[str] = None              # 종료 일시 (ISO 8601)
    location: Optional[str] = None              # 장소
    recommendedAction: Optional[str] = None     # 카테고리 맞춤 추천 다음 행동


def build_schedule_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 '일정/공지'로 분류되었습니다.
Android 앱이 일정/공지 카드를 그대로 렌더링하고 캘린더 초안을 만들 수 있도록 아래 필드를 추출하세요.

반드시 JSON 객체 하나만 출력하세요. 설명, 마크다운, 코드펜스는 출력하지 마세요.
텍스트에 근거가 없거나 확신이 낮은 값은 null로 두세요. 없는 정보를 추측하지 마세요.
전화번호, 이메일, 계좌번호, 주민등록번호 같은 민감정보는 복원하거나 추측하지 마세요.

오늘 날짜: {today_iso}
사용자 로케일: {locale}
'내일', '이번 주 금요일' 같은 상대 표현은 오늘 날짜 기준으로 계산하세요.

JSON 스키마:
{{
  "type": "schedule",
  "title": "카드 제목. 모집/행사/공지명을 40자 이내로, 없으면 1단계 제목 기반",
  "source": "AI 추출",
  "deadline": {{
    "label": "마감",
    "date": "마감일 YYYY-MM-DD, 없으면 null",
    "time": "마감 시각 HH:mm, 없으면 null",
    "displayText": "사람이 읽기 좋은 마감 표시. 예: '2026년 7월 10일 23:59', 없으면 null"
  }},
  "eventDate": {{
    "date": "행사/시작일 YYYY-MM-DD, 없으면 null",
    "dayOfWeek": "월 | 화 | 수 | 목 | 금 | 토 | 일, 없으면 null",
    "displayText": "사람이 읽기 좋은 일정 표시. 예: '2026년 7월 10일 (금)', 없으면 null"
  }},
  "location": "장소, 없으면 null",
  "calendarAction": {{
    "label": "구글 캘린더에 추가",
    "enabled": true
  }},
  "startAtIso": "캘린더 초안용 시작 일시 ISO 8601 문자열, 없으면 null",
  "endAtIso": "캘린더 초안용 종료 일시 ISO 8601 문자열, 없으면 null",
  "recommendedAction": "추천 다음 행동. 예: '7월 10일 마감 알림 설정'"
}}

작성 규칙:
- deadline은 신청/모집/제출 마감이 있을 때 채우세요.
- eventDate는 실제 행사일, 수업 시작일, 설명회 일시가 있을 때 채우세요.
- 같은 날짜가 마감일이자 행사일인 경우 둘 다 채울 수 있습니다.
- calendarAction.enabled는 startAtIso 또는 deadline.date 또는 eventDate.date 중 하나라도 있으면 true, 모두 없으면 false로 두세요.
- title이 애매하면 1단계 제목 "{base.title}"을 사용하세요.

텍스트:
\"\"\"
{text}
\"\"\"
"""
