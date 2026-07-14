import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.analysis.prompt import LlmAnalysis


class RestaurantMenuItem(BaseModel):
    """A menu item extracted from restaurant OCR text."""

    name: str
    price: Optional[int] = None
    currency: str = "KRW"

    @field_validator("price", mode="before")
    @classmethod
    def normalize_price(cls, value: object) -> Optional[int]:
        if value is None or isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else None
        return None


class RestaurantRecommendedAction(BaseModel):
    """A concrete user-facing recommendation for visiting or saving the place."""

    type: Literal["visit_time", "companion", "reservation", "budget", "map", "save", "other"] = "other"
    title: str
    description: Optional[str] = None


class RestaurantPlaceDetails(BaseModel):
    """Restaurant details that the Android app can store and render directly."""

    name: Optional[str] = None
    address: Optional[str] = None
    roadAddress: Optional[str] = None
    neighborhood: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    mapProvider: Optional[Literal["kakao", "naver", "google"]] = None
    mapProviderPlaceId: Optional[str] = None
    menus: list[RestaurantMenuItem] = Field(default_factory=list)
    estimatedPricePerPersonMin: Optional[int] = None
    estimatedPricePerPersonMax: Optional[int] = None
    tags: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    recommendedActions: list[RestaurantRecommendedAction] = Field(default_factory=list)

    @field_validator("estimatedPricePerPersonMin", "estimatedPricePerPersonMax", mode="before")
    @classmethod
    def normalize_price_range(cls, value: object) -> Optional[int]:
        if value is None or isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            digits = "".join(ch for ch in value if ch.isdigit())
            return int(digits) if digits else None
        return None

    @field_validator("tags", "features", mode="before")
    @classmethod
    def normalize_string_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


class RestaurantGroup(BaseModel):
    """Neighborhood group metadata for cards like '성수동 맛집'."""

    id: Optional[str] = None
    title: Optional[str] = None
    neighborhood: Optional[str] = None


class RestaurantExtraction(BaseModel):
    """One restaurant and the source screenshots that support it."""

    sourceRefs: list[str] = Field(default_factory=list)
    title: Optional[str] = None
    topicKey: Optional[str] = None
    summary: Optional[str] = None
    restaurant: RestaurantPlaceDetails = Field(default_factory=RestaurantPlaceDetails)
    group: Optional[RestaurantGroup] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needsUserReview: bool = False
    recommendedAction: Optional[str] = None

    @model_validator(mode="after")
    def fill_derived_flags(self) -> "RestaurantExtraction":
        restaurant = self.restaurant
        has_place_identity = bool(restaurant.name)
        has_location = bool(restaurant.address or restaurant.roadAddress or (restaurant.latitude and restaurant.longitude))

        if not has_place_identity or not has_location:
            self.needsUserReview = True
            self.confidence = min(self.confidence, 0.6)

        # LLM이 임의로 만든 id를 신뢰하지 않는다. 같은 동네는 항상 같은 그룹 id를 쓴다.
        neighborhood = _normalize_neighborhood(restaurant.neighborhood)
        if neighborhood:
            restaurant.neighborhood = neighborhood
            group_id = _slugify_korean_safe(f"{restaurant.neighborhood}-restaurant")
            self.group = RestaurantGroup(
                id=group_id,
                title=f"{restaurant.neighborhood} 맛집",
                neighborhood=restaurant.neighborhood,
            )

        return self


class RestaurantDetails(BaseModel):
    """Stage-2 result. BatchAnalyzer expands each entry into one MemoGroup."""

    restaurants: list[RestaurantExtraction] = Field(default_factory=list)


def build_restaurant_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""다음 OCR 텍스트는 '맛집' 카테고리로 분류되었습니다.
Android 앱이 맛집 상세 화면, 지도 화면, 동네 그룹 카드를 만들 수 있도록 장소 정보를 추출하세요.

반드시 JSON 객체 하나만 출력하세요. 설명, 마크다운, 코드펜스는 출력하지 마세요.
텍스트에 근거가 없거나 확신이 낮은 값은 null 또는 빈 배열로 두세요.
전화번호, 이메일, 계좌번호, 주민등록번호 같은 민감정보는 복원하거나 추측하지 마세요.

오늘 날짜: {today_iso}
사용자 로케일: {locale}

입력에는 각 원본 스크린샷이 [SOURCE S0] ... [/SOURCE] 형식으로 구분되어 있습니다.
각 SOURCE에 가게가 하나만 있으면 SOURCE마다 restaurants 항목을 하나씩 따로 만드세요.
한 SOURCE가 여러 가게의 메뉴나 가격을 비교하면 가게별로 나누지 말고 비교 메모 항목 하나만 만드세요.

JSON 스키마:
{{
  "restaurants": [
    {{
      "sourceRefs": ["S0"],
      "title": "가게명 중심의 40자 이내 메모 제목",
      "topicKey": null,
      "summary": "이 가게만 설명하는 1~2문장 요약",
      "restaurant": {{
        "name": "가게명 또는 장소명, 없으면 null",
        "address": "주소, 없으면 null",
        "roadAddress": "도로명 주소, 없으면 null",
        "neighborhood": "동네명 예: 성수동, 없으면 null",
        "latitude": null,
        "longitude": null,
        "mapProvider": null,
        "mapProviderPlaceId": null,
        "menus": [{{"name": "에그베네딕트", "price": 18000, "currency": "KRW"}}],
        "estimatedPricePerPersonMin": 20000,
        "estimatedPricePerPersonMax": 30000,
        "tags": ["데이트", "브런치", "혼밥"],
        "features": ["주말 웨이팅 가능성 높음", "평일 방문 추천"],
        "recommendedActions": [
          {{
            "type": "visit_time",
            "title": "평일 오전 방문",
            "description": "웨이팅을 줄이려면 평일 오픈 시간대 방문을 추천"
          }}
        ]
      }},
      "group": {{
        "id": "seongsu-restaurant",
        "title": "성수동 맛집",
        "neighborhood": "성수동"
      }},
      "confidence": 0.86,
      "needsUserReview": false,
      "recommendedAction": "지도에서 위치 확인 후 방문 리스트에 저장"
    }}
  ]
}}

추출 규칙:
- 가게명은 상호명으로 보이는 가장 구체적인 이름을 사용하세요.
- sourceRefs에는 해당 가게의 근거가 실제로 존재하는 SOURCE 식별자만 정확히 복사하세요.
- 모든 SOURCE를 최소 한 항목에 포함하고, 존재하지 않는 SOURCE 식별자는 만들지 마세요.
- 가게가 하나뿐인 SOURCE는 다른 SOURCE와 주제나 가게가 같아도 반드시 별도 항목으로 만드세요.
- 단일 가게 항목의 topicKey는 null로 두고 title과 summary에 다른 가게 정보를 섞지 마세요.
- 한 SOURCE에 여러 가게의 메뉴·가격 비교가 있으면 비교 전체를 항목 하나로 만드세요.
- 비교 항목은 title을 비교 주제 중심으로 쓰고 topicKey를 짧게 정규화하세요. 예: "수박주스 비교".
- 동네명은 숫자를 제거한 기본형으로 정규화하세요. 예: "상도1동" -> "상도동".
- 비교 항목의 restaurant.name, 주소, 좌표, group은 null로 두세요.
- 비교 항목의 menus에는 "가게명 · 메뉴명" 형식으로 가게를 구분하여 모든 비교 메뉴를 넣으세요.
- 여러 SOURCE가 같은 비교 주제를 이어서 다루면 하나의 비교 항목에 SOURCE들을 모두 포함하세요.
- 같은 비교 주제가 여러 항목으로 불가피하게 나뉘면 topicKey를 완전히 동일하게 작성하세요.
- 메뉴 가격은 정수 원화로 변환하세요. 예: "18,000원" -> 18000.
- 1인 예상 금액은 메뉴/가격 근거가 있을 때만 채우세요.
- 태그는 최대 5개, features는 최대 5개, recommendedActions는 최대 3개로 제한하세요.
- 주소가 없으면 needsUserReview를 true로 두세요.
- 좌표와 지도 제공자 정보는 서버가 Kakao API로 확인합니다. latitude, longitude,
  mapProvider, mapProviderPlaceId는 OCR 텍스트에 값이 있어도 반드시 null로 출력하세요.
- neighborhood가 있으면 group을 만들고, title은 "<동네명> 맛집" 형식으로 작성하세요.
- 정보가 가게명뿐이면 name만 채우고 나머지는 null/빈 배열, confidence는 0.6 이하로 두세요.

OCR 텍스트:
\"\"\"
{text}
\"\"\"
"""


def _slugify_korean_safe(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    return "".join(ch for ch in normalized if ch.isalnum() or ch == "-") or "restaurant-group"


def _normalize_neighborhood(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = re.sub(r"\d+(?=(동|가|읍|면|리)$)", "", value.strip())
    return normalized or None
