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


class RestaurantDetails(BaseModel):
    """Stage-2 restaurant analysis payload returned inside AnalyzeResponse.details."""

    restaurant: RestaurantPlaceDetails = Field(default_factory=RestaurantPlaceDetails)
    group: Optional[RestaurantGroup] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needsUserReview: bool = False
    recommendedAction: Optional[str] = None

    @model_validator(mode="after")
    def fill_derived_flags(self) -> "RestaurantDetails":
        restaurant = self.restaurant
        has_place_identity = bool(restaurant.name)
        has_location = bool(restaurant.address or restaurant.roadAddress or (restaurant.latitude and restaurant.longitude))

        if not has_place_identity or not has_location:
            self.needsUserReview = True
            self.confidence = min(self.confidence, 0.6)

        if self.group is None and restaurant.neighborhood:
            group_id = _slugify_korean_safe(f"{restaurant.neighborhood}-restaurant")
            self.group = RestaurantGroup(
                id=group_id,
                title=f"{restaurant.neighborhood} 맛집",
                neighborhood=restaurant.neighborhood,
            )

        return self


def build_restaurant_prompt(text: str, base: LlmAnalysis, locale: str, today_iso: str) -> str:
    return f"""다음 OCR 텍스트는 '맛집' 카테고리로 분류되었습니다.
Android 앱이 맛집 상세 화면, 지도 화면, 동네 그룹 카드를 만들 수 있도록 장소 정보를 추출하세요.

반드시 JSON 객체 하나만 출력하세요. 설명, 마크다운, 코드펜스는 출력하지 마세요.
텍스트에 근거가 없거나 확신이 낮은 값은 null 또는 빈 배열로 두세요.
전화번호, 이메일, 계좌번호, 주민등록번호 같은 민감정보는 복원하거나 추측하지 마세요.

오늘 날짜: {today_iso}
사용자 로케일: {locale}

JSON 스키마:
{{
  "restaurant": {{
    "name": "가게명 또는 장소명, 없으면 null",
    "address": "주소, 없으면 null",
    "roadAddress": "도로명 주소, 없으면 null",
    "neighborhood": "동네명 예: 성수동, 없으면 null",
    "latitude": null,
    "longitude": null,
    "mapProvider": null,
    "mapProviderPlaceId": null,
    "menus": [
      {{"name": "에그베네딕트", "price": 18000, "currency": "KRW"}}
    ],
    "estimatedPricePerPersonMin": 20000,
    "estimatedPricePerPersonMax": 30000,
    "tags": ["데이트", "브런치", "혼밥"],
    "features": ["주말 웨이팅 가능성 높음", "평일 방문 추천"],
    "recommendedActions": [
      {{
        "type": "visit_time",
        "title": "평일 오전 방문",
        "description": "웨이팅을 줄이려면 평일 오픈 시간대 방문을 추천"
      }},
      {{
        "type": "companion",
        "title": "데이트 또는 친구와 방문",
        "description": "브런치와 카페 메뉴가 있어 가벼운 약속에 적합"
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

추출 규칙:
- 가게명은 상호명으로 보이는 가장 구체적인 이름을 사용하세요.
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
