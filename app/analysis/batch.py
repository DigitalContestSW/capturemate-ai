import logging

from app.analysis.grouping import group_indices
from app.analysis.preprocess import preprocess_for_embedding
from app.analysis.service import AnalysisService, restore_in_response
from app.config import Settings, settings
from app.embeddings.base import EmbeddingClient, EmbeddingError
from app.embeddings.factory import build_embedding_client
from app.models import AnalyzeBatchItem, AnalyzeResponse, MemoGroup
from app.privacy import mask_text

logger = logging.getLogger("capturemate.batch")

# 임베딩 입력이 너무 길면 토큰 한계에 걸릴 수 있어 앞부분만 사용(그룹핑 판단엔 충분).
_EMBED_MAX_CHARS = 2000


class BatchAnalyzer:
    """하루치 스크린샷을 배치로 받아 유사한 것끼리 묶고, 그룹당 하나의 메모를 만든다.

    흐름: 서버측 마스킹 -> 임베딩 -> (cosine × 시간가중) 그룹핑 -> 그룹별 LLM 분석.
    개별 항목마다 LLM을 돌리지 않고 '그룹당 1회'만 분석해 비용/시간을 아낀다.
    """

    def __init__(
        self,
        analysis_service: AnalysisService | None = None,
        embedding_client: EmbeddingClient | None = None,
        config: Settings = settings,
    ) -> None:
        self._config = config
        self._analysis = analysis_service or AnalysisService(config=config)
        self._embedding = (
            embedding_client if embedding_client is not None else build_embedding_client(config)
        )

    def analyze_batch(self, items: list[AnalyzeBatchItem], locale: str) -> list[MemoGroup]:
        if not items:
            return []

        # ① 서버측 마스킹 (+ 복원 매핑). namespace로 항목 간 토큰 충돌 방지([PHONE_0_1] 등)
        mask_results = [mask_text(item.maskedText, namespace=f"{i}_") for i, item in enumerate(items)]
        masked = [mr.text for mr in mask_results]
        timestamps = [item.capturedAt for item in items]

        # ② 유사 항목 그룹핑 (임베딩 불가 시 각 항목을 개별 그룹으로)
        groups_idx = self._group(masked, timestamps)

        # ③ 그룹별 분석. 맛집은 장소별 결과를 다시 여러 MemoGroup으로 펼친다.
        result: list[MemoGroup] = []
        for idx_list in groups_idx:
            merged = _merge_with_source_refs(masked, idx_list)
            merged_mapping: dict[str, str] = {}
            for i in idx_list:
                merged_mapping.update(mask_results[i].mapping)

            analysis = self._analysis.analyze(merged, locale)
            if analysis.category == "restaurant":
                restaurant_groups = _split_restaurant_analysis(analysis, idx_list, items)
                if restaurant_groups is None:
                    logger.warning("맛집 SOURCE 매핑 검증 실패, 그룹을 개별 재분석")
                    result.extend(
                        self._analyze_restaurant_sources_individually(
                            idx_list, items, masked, mask_results, locale
                        )
                    )
                else:
                    result.extend(
                        MemoGroup(
                            memberClientIds=group.memberClientIds,
                            analysis=restore_in_response(group.analysis, merged_mapping),
                        )
                        for group in restaurant_groups
                    )
                continue

            analysis = restore_in_response(analysis, merged_mapping)
            result.append(
                MemoGroup(
                    memberClientIds=[items[i].clientId for i in idx_list],
                    analysis=analysis,
                )
            )
        return result

    def _analyze_restaurant_sources_individually(
        self,
        idx_list: list[int],
        items: list[AnalyzeBatchItem],
        masked: list[str],
        mask_results: list,
        locale: str,
    ) -> list[MemoGroup]:
        result: list[MemoGroup] = []
        for item_index in idx_list:
            analysis = self._analysis.analyze(
                _merge_with_source_refs(masked, [item_index]), locale
            )
            split = (
                _split_restaurant_analysis(analysis, [item_index], items)
                if analysis.category == "restaurant"
                else None
            )
            if split is None:
                split = [
                    MemoGroup(
                        memberClientIds=[items[item_index].clientId],
                        analysis=analysis,
                    )
                ]
            result.extend(
                MemoGroup(
                    memberClientIds=group.memberClientIds,
                    analysis=restore_in_response(group.analysis, mask_results[item_index].mapping),
                )
                for group in split
            )
        return result

    def _group(self, masked: list[str], timestamps: list[int | None]) -> list[list[int]]:
        if self._embedding is None:
            return [[i] for i in range(len(masked))]  # 그룹핑 없이 개별 처리(안전한 기본값)

        # 임베딩(그룹핑)용으로만 전처리(노이즈 제거). LLM 분석엔 원본 masked를 쓴다.
        # 전처리로 전부 비면 원본으로 폴백(빈 문자열 임베딩 방지).
        embed_inputs = [(preprocess_for_embedding(m) or m)[:_EMBED_MAX_CHARS] for m in masked]
        try:
            embeddings = self._embedding.embed(embed_inputs)
        except EmbeddingError as exc:
            logger.warning("embedding 실패, 그룹핑 생략: %s", type(exc).__name__)
            return [[i] for i in range(len(masked))]

        # 임베딩 수 != 항목 수면 그룹핑이 항목을 누락시킬 수 있으므로, 안전하게 개별 처리.
        if len(embeddings) != len(masked):
            logger.warning(
                "embedding 수 불일치(%d != %d), 그룹핑 생략",
                len(embeddings),
                len(masked),
            )
            return [[i] for i in range(len(masked))]

        return group_indices(
            embeddings,
            timestamps,
            threshold=self._config.grouping_threshold,
            tau_seconds=self._config.grouping_tau_seconds,
            w_min=self._config.grouping_w_min,
        )


def _merge_with_source_refs(masked: list[str], idx_list: list[int]) -> str:
    parts = []
    for source_index, item_index in enumerate(idx_list):
        if masked[item_index]:
            parts.append(f"[SOURCE S{source_index}]\n{masked[item_index]}\n[/SOURCE]")
    return "\n\n".join(parts)


def _split_restaurant_analysis(
    analysis: AnalyzeResponse,
    idx_list: list[int],
    items: list[AnalyzeBatchItem],
) -> list[MemoGroup] | None:
    details = analysis.details or {}
    extracted = details.get("restaurants")
    if not isinstance(extracted, list) or not extracted:
        return None

    valid_refs = {f"S{i}" for i in range(len(idx_list))}
    covered_refs: set[str] = set()
    groups: list[MemoGroup] = []

    for entry in extracted:
        if not isinstance(entry, dict):
            return None
        raw_refs = entry.get("sourceRefs")
        if not isinstance(raw_refs, list) or not raw_refs:
            return None
        refs = list(dict.fromkeys(ref for ref in raw_refs if isinstance(ref, str)))
        if not refs or any(ref not in valid_refs for ref in refs):
            return None
        covered_refs.update(refs)

        place_details = {
            key: value
            for key, value in entry.items()
            if key not in {"sourceRefs", "title", "summary", "recommendedAction"}
        }
        restaurant = place_details.get("restaurant")
        restaurant_name = restaurant.get("name") if isinstance(restaurant, dict) else None
        title = entry.get("title") or restaurant_name or analysis.title
        summary = entry.get("summary") or analysis.summary
        recommended_action = entry.get("recommendedAction") or analysis.recommendedAction
        member_ids = [
            items[idx_list[int(ref[1:])]].clientId
            for ref in refs
        ]
        groups.append(
            MemoGroup(
                memberClientIds=member_ids,
                analysis=analysis.model_copy(
                    update={
                        "title": str(title)[:40],
                        "summary": str(summary),
                        "recommendedAction": recommended_action,
                        "details": place_details,
                    }
                ),
            )
        )

    # 누락된 원본이 있으면 사진을 임의의 가게에 붙이지 않고 개별 재분석한다.
    if covered_refs != valid_refs:
        return None
    return groups
