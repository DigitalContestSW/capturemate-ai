from typing import Optional

from pydantic import BaseModel

from app.analysis.prompt import LlmAnalysis

# ── 맛집(restaurant) 담당자가 채워 확장하는 파일 ────────────────────────────
# 목표 액션: 외부 지도 연동 + 리마인더
# LLM은 지도에서 검색할 장소 정보를 뽑는다. 지도 열기/알림 예약은 안드로이드 몫.


class RestaurantDetails(BaseModel):
    """지도 연동/리마인더에 필요한 필드. (담당자가 자유롭게 추가/수정)"""

    placeName: Optional[str] = None      # 가게/장소명
    address: Optional[str] = None        # 주소
    reminderAtIso: Optional[str] = None  # 방문 알림 시각 (ISO 8601)


def build_restaurant_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 '맛집'으로 분류되었습니다. 지도 연동을 위해 장소 정보를
추출하세요. JSON 객체 하나만 출력(설명·코드펜스 없이).

JSON 스키마:
{{
  "placeName": "가게/장소명, 없으면 null",
  "address": "주소, 없으면 null",
  "reminderAtIso": "방문 알림 시각 ISO 8601 문자열, 없으면 null"
}}

텍스트:
\"\"\"
{text}
\"\"\"
"""
