from typing import Optional

from pydantic import BaseModel

# 카테고리의 단일 원천(single source of truth). 프롬프트와 폴백이 함께 공유한다.
# 실제 서비스 카테고리 5종. 안드로이드 CaptureCategory enum과 매핑:
#   schedule  -> 일정
#   study     -> 학습
#   life_info -> 생활정보
#   restaurant-> 맛집
#   shopping  -> 쇼핑 (분류만 지원, 2단계 세부 추출은 아직 미구현)
CATEGORIES = ["schedule", "study", "life_info", "restaurant", "shopping"]

# 위 5종 중 어디에도 확실히 속하지 않을 때 쓰는 안전용 값(안드로이드 Unknown).
UNKNOWN_CATEGORY = "unknown"

# 검증/클램프에 사용할 허용 집합.
ALLOWED_CATEGORIES = {*CATEGORIES, UNKNOWN_CATEGORY}


class LlmAnalysis(BaseModel):
    """LLM으로부터 돌려받기를 기대하는 1단계(분류 + 유용성) 결과.

    Pydantic이 알 수 없는 키는 무시하고 누락된 키는 기본값으로 채우므로,
    부분적이거나 지저분한 LLM 응답도 요청을 깨뜨리지 않고 검증을 통과한다.

    `isUseful` 기본값은 True — 판단이 애매하면 '남기는' 쪽으로 기운다
    (유용한 걸 놓치는 것이 잡담 하나 남기는 것보다 나쁘므로).
    """

    title: str = ""
    summary: str = ""
    category: str = UNKNOWN_CATEGORY
    isUseful: bool = True
    usefulReason: str = ""
    recommendedAction: Optional[str] = None
    reminderAtIso: Optional[str] = None


def build_classify_prompt(masked_text: str, locale: str, today_iso: str) -> str:
    categories = ", ".join(CATEGORIES)
    return f"""당신은 사용자의 스크린샷에서 추출·마스킹된 텍스트를 분석하는 어시스턴트입니다.

[민감정보 토큰 규칙]
대괄호 토큰([PHONE_1], [EMAIL_2] 등)은 민감정보가 가려진 자리입니다.
- 실제 값을 추측·복원하지 마세요.
- 다만 메모에 담아야 할 정보(예: 문의처, 예약 연락처)라면 해당 토큰을 결과 텍스트에
  '그대로' 포함하세요. 예) "문의: [PHONE_1]"
- [RRN], [CARD] 같은 초민감 토큰은 결과에 절대 포함하지 마세요.

[유용성 판단]
먼저 이 스크린샷이 '저장할 가치가 있는 정보'인지 판단하세요.
- isUseful=true: 나중에 행동하거나 참고할 정보가 있음
  (마감일/일정, 방문할 장소, 구매·신청할 것, 학습/참고 자료, 문의처 등)
- isUseful=false: 다시 볼 이유가 없음
  (일상 대화/잡담, 밈/유머, 개인 사진, 단순 UI·설정 화면, 감정 표현 위주)
- 판단이 애매하면 isUseful=true (놓치는 것보다 남기는 게 낫다)

오늘 날짜: {today_iso}
사용자 로케일: {locale}

아래 텍스트를 분석해 JSON 객체 '하나만' 출력하세요. 설명·마크다운·코드펜스 없이 순수 JSON만.

카테고리는 반드시 다음 중 하나입니다: {categories}
- schedule: 일정, 공지, 회의, 예약, 마감일, 신청/모집 마감 등 '기한이 있는 안내'
- study: 시험, 강의, 학습 자료, 복습, 과제
- life_info: 정책/제도, 금융/적금, 교통, 공공 정보, 생활 꿀팁 등 일상 생활정보
- restaurant: 맛집, 카페, 메뉴, 방문할 장소
- shopping: 상품 구매, 쇼핑, 할인/쿠폰/특가, 위시리스트
위 5개 중 어디에도 확실히 속하지 않으면 "{UNKNOWN_CATEGORY}"를 사용하세요.

JSON 스키마:
{{
  "isUseful": true 또는 false,
  "usefulReason": "유용성 판단 근거 한 줄",
  "title": "40자 이내 제목",
  "summary": "1~2문장 요약",
  "category": "{categories} 중 하나 (또는 {UNKNOWN_CATEGORY})",
  "recommendedAction": "추천하는 다음 행동 (없으면 null)",
  "reminderAtIso": "마감일/일시가 있으면 ISO 8601 문자열(예: 2026-07-10T09:00:00), 없으면 null"
}}

reminderAtIso는 텍스트에 실제 마감일이나 일시가 있을 때만 채우세요.
'내일', '이번 주 금요일', '3일 후' 같은 표현은 오늘 날짜({today_iso}) 기준으로 계산하세요.

분석할 텍스트:
\"\"\"
{masked_text}
\"\"\"
"""
