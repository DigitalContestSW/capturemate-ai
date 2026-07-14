import json
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger("capturemate.kakao_local")

_KEYWORD_SEARCH_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
_RESTAURANT_CATEGORY_CODES = ("FD6", "CE7")


@dataclass(frozen=True)
class KakaoPlace:
    id: str | None
    name: str | None
    address: str | None
    road_address: str | None
    longitude: float | None
    latitude: float | None
    category_name: str | None


class KakaoLocalClient:
    def __init__(self, rest_api_key: str, timeout_seconds: float = 3.0) -> None:
        self._rest_api_key = rest_api_key
        self._timeout_seconds = timeout_seconds

    def search_best_place(
        self,
        name: str | None,
        address: str | None = None,
        neighborhood: str | None = None,
    ) -> KakaoPlace | None:
        cleaned_name = _clean_string(name)
        if not cleaned_name:
            return None

        # 최초 검색은 식당명만 사용한다. 잘못 추출된 주소가 검색을 오염시키는 것을 막는다.
        candidates = self._search_candidates(cleaned_name)
        if len(candidates) == 1:
            return candidates[0]

        # 동명 지점이 여러 개이거나 첫 검색이 비었을 때만 동네명을 추가하여 다시 검색한다.
        cleaned_neighborhood = _clean_string(neighborhood)
        if cleaned_neighborhood and (not candidates or len(candidates) > 1):
            refined = self._search_candidates(f"{cleaned_name} {cleaned_neighborhood}")
            if len(refined) == 1:
                return refined[0]

        return None

    def _search_candidates(self, query: str) -> list[KakaoPlace]:
        candidates: list[KakaoPlace] = []
        for category_code in _RESTAURANT_CATEGORY_CODES:
            candidates.extend(self._search_keyword(query, category_code=category_code))
            if candidates:
                break

        if not candidates:
            candidates.extend(self._search_keyword(query, category_code=None))

        return candidates

    def _search_keyword(self, query: str, category_code: str | None) -> list[KakaoPlace]:
        params: dict[str, str | int] = {
            "query": query,
            "size": 5,
            "page": 1,
        }
        if category_code:
            params["category_group_code"] = category_code

        url = f"{_KEYWORD_SEARCH_URL}?{urlencode(params)}"
        request = Request(
            url,
            headers={"Authorization": f"KakaoAK {self._rest_api_key}"},
            method="GET",
        )

        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            logger.warning("Kakao Local keyword search failed: HTTP %s %s", exc.code, body)
            return []
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("Kakao Local keyword search failed: %s", type(exc).__name__)
            return []

        documents = payload.get("documents", [])
        if not isinstance(documents, list):
            return []

        return [_place_from_document(document) for document in documents if isinstance(document, dict)]


def enrich_restaurant_details_with_kakao(details: dict[str, Any], client: KakaoLocalClient) -> dict[str, Any]:
    restaurant = details.get("restaurant")
    if not isinstance(restaurant, dict):
        return details

    name = _clean_string(restaurant.get("name"))
    address = _clean_string(restaurant.get("address") or restaurant.get("roadAddress"))
    neighborhood = _clean_string(restaurant.get("neighborhood"))
    if not name:
        return details

    place = client.search_best_place(name=name, address=address, neighborhood=neighborhood)
    if place is None:
        details["needsUserReview"] = True
        details["confidence"] = min(float(details.get("confidence") or 0.5), 0.6)
        return details

    restaurant["name"] = restaurant.get("name") or place.name
    restaurant["address"] = restaurant.get("address") or place.address
    restaurant["roadAddress"] = restaurant.get("roadAddress") or place.road_address
    # 좌표는 LLM 값이 아니라 Kakao 검색 결과를 기준으로 사용한다.
    restaurant["latitude"] = place.latitude
    restaurant["longitude"] = place.longitude
    restaurant["mapProvider"] = "kakao"
    restaurant["mapProviderPlaceId"] = place.id

    if not restaurant.get("neighborhood"):
        restaurant["neighborhood"] = _extract_neighborhood(place.address) or _extract_neighborhood(place.road_address)

    if restaurant.get("address") or restaurant.get("roadAddress"):
        details["needsUserReview"] = False
        details["confidence"] = max(float(details.get("confidence") or 0.5), 0.75)

    # Kakao로 보강된 최종 동네를 기준으로 그룹을 다시 정규화한다.
    if restaurant.get("neighborhood"):
        neighborhood_value = restaurant["neighborhood"]
        details["group"] = {
            "id": _slugify_korean_safe(f"{neighborhood_value}-restaurant"),
            "title": f"{neighborhood_value} 맛집",
            "neighborhood": neighborhood_value,
        }

    return details


def _place_from_document(document: dict[str, Any]) -> KakaoPlace:
    return KakaoPlace(
        id=_clean_string(document.get("id")),
        name=_clean_string(document.get("place_name")),
        address=_clean_string(document.get("address_name")),
        road_address=_clean_string(document.get("road_address_name")),
        longitude=_to_float(document.get("x")),
        latitude=_to_float(document.get("y")),
        category_name=_clean_string(document.get("category_name")),
    )

def _extract_neighborhood(address: str | None) -> str | None:
    if not address:
        return None
    for token in address.split():
        if re.search(r"(동|가|읍|면|리)$", token) and not token.endswith(("시", "구")):
            return token
    return None


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _slugify_korean_safe(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    return "".join(ch for ch in normalized if ch.isalnum() or ch == "-") or "restaurant-group"
