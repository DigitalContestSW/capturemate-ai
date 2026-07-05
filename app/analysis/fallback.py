from app.analysis.prompt import UNKNOWN_CATEGORY, LlmAnalysis

# LLM을 쓸 수 없거나 모든 재시도가 실패했을 때 사용하는 키워드 분류기.
# 한국어 키워드를 포함해 실제 ko-KR 캡처에서도 자연스럽게 동작하도록 했다.
# 카테고리는 실제 서비스 5종 + 안전용 unknown. (먼저 매칭되는 카테고리가 선택되므로
# 겹칠 수 있는 shopping을 life_info보다 앞에 둔다.)
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "schedule": ["일정", "공지", "회의", "예약", "마감", "미팅", "신청", "모집", "채용", "지원", "면접"],
    "study": ["시험", "강의", "복습", "과제", "학습", "수업", "레포트"],
    "restaurant": ["맛집", "카페", "메뉴", "식당", "맛있", "방문"],
    "shopping": ["쇼핑", "구매", "할인", "쿠폰", "특가", "세일", "장바구니", "위시리스트", "상품"],
    "life_info": ["적금", "금융", "정책", "교통", "공공", "혜택", "생활"],
}

_ACTIONS: dict[str, str] = {
    "schedule": "캘린더에 일정 추가",
    "study": "학습 노트로 저장",
    "life_info": "생활정보로 저장",
    "restaurant": "가볼 장소로 저장",
    "shopping": "쇼핑 목록에 저장",
    UNKNOWN_CATEGORY: "메모로 저장",
}


def fallback_analysis(text: str) -> LlmAnalysis:
    lowered = text.lower()

    category = UNKNOWN_CATEGORY
    for candidate, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            category = candidate
            break

    stripped = text.strip()
    title = stripped.splitlines()[0][:40] if stripped else "New capture"
    summary = " ".join(text.split())[:120] or "No content to summarize."

    return LlmAnalysis(
        title=title or "New capture",
        summary=summary,
        category=category,
        recommendedAction=_ACTIONS.get(category),
        reminderAtIso=None,
    )
