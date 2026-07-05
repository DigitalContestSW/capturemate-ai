from typing import Optional

from pydantic import BaseModel

from app.analysis.prompt import LlmAnalysis

# ── 맛집(restaurant) 담당자가 채워 확장하는 파일 ────────────────────────────
# 와이어프레임: 장소 정보 (장소, 메뉴, 예상 금액, 태그 등)
#   - LLM은 텍스트에서 장소 정보를 추출. 좌표/상세 주소 보완은 안드로이드 지도 API 몫.


class RestaurantDetails(BaseModel):
    """맛집 세부: 지도 연동 및 카드 표시용 장소 정보. (담당자가 자유롭게 추가/수정)"""

    placeName: Optional[str] = None             # 장소명
    address: Optional[str] = None               # 주소(텍스트에 있으면; 없으면 지도 API가 보완)
    menu: list[str] = []                        # 대표 메뉴
    estimatedPrice: Optional[str] = None        # 예상 금액(예: "1인 2~3만원")
    tags: list[str] = []                        # 태그(분위기/종류 등, 예: "브런치", "데이트")
    recommendedAction: Optional[str] = None     # 카테고리 맞춤 추천 다음 행동


def build_restaurant_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""이 텍스트는 '맛집'으로 분류되었습니다. 지도 연동과 카드 표시를 위해
장소 정보를 추출하세요. JSON 객체 하나만 출력(설명·코드펜스 없이).

JSON 스키마:
{{
  "placeName": "장소/가게명, 없으면 null",
  "address": "주소, 없으면 null",
  "menu": ["대표 메뉴1", "대표 메뉴2"],
  "estimatedPrice": "예상 금액(예: 1인 2~3만원), 없으면 null",
  "tags": ["태그1", "태그2"],
  "recommendedAction": "추천 다음 행동 (예: 지도에서 위치 보기, 방문 리스트에 저장)"
}}

menu/tags는 텍스트에 근거가 있을 때만 채우고, 없으면 빈 배열로 두세요.

텍스트:
\"\"\"
{text}
\"\"\"
"""
