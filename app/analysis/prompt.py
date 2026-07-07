from typing import Optional

from pydantic import BaseModel

# 카테고리의 단일 원천(single source of truth). 프롬프트와 폴백이 함께 공유한다.
# 실제 서비스 카테고리 4종. 안드로이드 CaptureCategory enum과 매핑:
#   schedule  -> Schedule   (일정/공지)
#   study     -> Study      (학습)
#   life_info -> LifeInfo   (생활정보)
#   restaurant-> Restaurant (맛집)
CATEGORIES = ["schedule", "study", "life_info", "restaurant"]

# 위 4종 중 어디에도 확실히 속하지 않을 때 쓰는 안전용 값(안드로이드 Unknown).
# LLM이 억지로 오분류하지 않도록 두는 탈출구일 뿐, 정식 5번째 카테고리는 아니다.
UNKNOWN_CATEGORY = "unknown"

# 검증/클램프에 사용할 허용 집합.
ALLOWED_CATEGORIES = {*CATEGORIES, UNKNOWN_CATEGORY}


class LlmAnalysis(BaseModel):
    """LLM으로부터 돌려받기를 기대하는 구조화 결과.

    `AnalyzeResponse`(공개 API 계약)와 일부러 분리했다. 이건 모델이 생성하는
    '내부' 형태다. Pydantic이 알 수 없는 키는 무시하고 누락된 키는 기본값으로
    채우므로, 부분적이거나 지저분한 LLM 응답도 요청을 깨뜨리지 않고 검증을 통과한다.

    `reminderAtIso`가 epoch ms가 아니라 ISO 8601 문자열인 점에 주의. LLM에게서
    날짜 문자열을 받아 Python에서 변환하는 편이, 모델에게 타임스탬프 산술을
    시키는 것보다 훨씬 안정적이다.
    """

    title: str = ""
    summary: str = ""
    category: str = UNKNOWN_CATEGORY
    recommendedAction: Optional[str] = None
    reminderAtIso: Optional[str] = None


# 1단계(분류) 프롬프트. 카테고리 + 어느 카테고리든 쓸 수 있는 기본 설명을 만든다.
# 2단계(카테고리별 세부 추출)는 app/analysis/categories/ 아래에 있다.
def build_classify_prompt(masked_text: str, locale: str, today_iso: str) -> str:
    categories = ", ".join(CATEGORIES)
    return f"""당신은 사용자의 스크린샷에서 추출·마스킹된 텍스트를 분석하는 어시스턴트입니다.
[EMAIL], [PHONE], [RRN], [CARD] 같은 대괄호 토큰은 민감정보가 가려진 자리입니다. 복원하려 하지 마세요.

오늘 날짜: {today_iso}
사용자 로케일: {locale}

아래 텍스트를 분석해 JSON 객체 '하나만' 출력하세요. 설명·마크다운·코드펜스 없이 순수 JSON만.

카테고리는 반드시 다음 중 하나입니다: {categories}
- schedule: 일정, 행사, 공지, 회의, 예약, 예매 오픈, 마감일, 신청/모집 마감 등 '날짜나 시간이 있는 안내'
- study: 시험, 강의, 학습 자료, 복습, 과제
- life_info: 쿠폰, 할인, 쇼핑, 결제/영수증, 예매, 교통 등 일상 생활정보
- restaurant: 맛집, 카페, 메뉴, 방문할 장소
위 4개 중 어디에도 확실히 속하지 않으면 "{UNKNOWN_CATEGORY}"를 사용하세요.

JSON 스키마:
{{
  "title": "40자 이내 제목",
  "summary": "1~2문장 요약",
  "category": "{categories} 중 하나 (또는 {UNKNOWN_CATEGORY})",
  "recommendedAction": "추천하는 다음 행동 (없으면 null)",
  "reminderAtIso": "마감일/일시가 있으면 ISO 8601 문자열(예: 2026-07-10T09:00:00), 없으면 null"
}}

schedule로 판단한 경우 summary는 메모에서 바로 읽기 좋은 한 문장으로 쓰세요.
예: "7월 3일~13일 개최, 주요 상영작과 예매 오픈 일정 캡처"

reminderAtIso는 텍스트에 실제 마감일이나 일시가 있을 때만 채우세요.
'내일', '이번 주 금요일', '3일 후' 같은 표현은 오늘 날짜({today_iso}) 기준으로 계산하세요.

분석할 텍스트:
\"\"\"
{masked_text}
\"\"\"
"""
