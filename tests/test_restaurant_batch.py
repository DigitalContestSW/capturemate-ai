import unittest

from app.analysis.batch import _split_restaurant_analysis
from app.models import AnalyzeBatchItem, AnalyzeResponse


def _analysis(restaurants: list[dict]) -> AnalyzeResponse:
    return AnalyzeResponse(
        title="신논현 직장인 맛집",
        summary="두 맛집을 소개합니다.",
        category="restaurant",
        details={"restaurants": restaurants},
    )


class RestaurantBatchSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.items = [
            AnalyzeBatchItem(clientId="capture-kyokoko", maskedText="쿄코코"),
            AnalyzeBatchItem(clientId="capture-hamsura", maskedText="함수라"),
        ]

    def test_splits_different_restaurants_and_sources(self) -> None:
        result = _split_restaurant_analysis(
            _analysis(
                [
                    {"sourceRefs": ["S0"], "title": "쿄코코", "summary": "쿄코코 상세", "restaurant": {"name": "쿄코코"}},
                    {"sourceRefs": ["S1"], "title": "함수라", "summary": "함수라 상세", "restaurant": {"name": "함수라"}},
                ]
            ),
            [0, 1],
            self.items,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(["capture-kyokoko"], result[0].memberClientIds)
        self.assertEqual("쿄코코", result[0].analysis.title)
        self.assertEqual(["capture-hamsura"], result[1].memberClientIds)
        self.assertEqual("함수라", result[1].analysis.title)
        self.assertNotIn("restaurants", result[0].analysis.details or {})

    def test_rejects_unknown_source(self) -> None:
        result = _split_restaurant_analysis(
            _analysis([{"sourceRefs": ["S7"], "restaurant": {"name": "쿄코코"}}]),
            [0, 1],
            self.items,
        )
        self.assertIsNone(result)

    def test_rejects_unassigned_source(self) -> None:
        result = _split_restaurant_analysis(
            _analysis([{"sourceRefs": ["S0"], "restaurant": {"name": "쿄코코"}}]),
            [0, 1],
            self.items,
        )
        self.assertIsNone(result)

    def test_allows_one_source_to_support_multiple_restaurants(self) -> None:
        result = _split_restaurant_analysis(
            _analysis(
                [
                    {"sourceRefs": ["S0"], "restaurant": {"name": "쿄코코"}},
                    {"sourceRefs": ["S0"], "restaurant": {"name": "함수라"}},
                ]
            ),
            [0],
            self.items[:1],
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(2, len(result))
        self.assertTrue(all(group.memberClientIds == ["capture-kyokoko"] for group in result))


if __name__ == "__main__":
    unittest.main()
