import logging
import time

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
            logger.info("배치 분석: 항목 없음 -> 빈 결과")
            return []

        started = time.perf_counter()
        logger.info("배치 분석 시작: items=%d", len(items))

        # ① 서버측 마스킹 (+ 복원 매핑). namespace로 항목 간 토큰 충돌 방지([PHONE_0_1] 등)
        mask_started = time.perf_counter()
        mask_results = [mask_text(item.maskedText, namespace=f"{i}_") for i, item in enumerate(items)]
        masked = [mr.text for mr in mask_results]
        timestamps = [item.capturedAt for item in items]
        masked_tokens = sum(len(mr.mapping) for mr in mask_results)
        logger.debug(
            "① 마스킹 완료: %d건, 소프트토큰 %d개 (%.0fms)",
            len(masked),
            masked_tokens,
            (time.perf_counter() - mask_started) * 1000,
        )

        # ② 유사 항목 그룹핑 (임베딩 불가 시 각 항목을 개별 그룹으로)
        groups_idx = self._group(masked, timestamps)
        logger.info(
            "② 그룹핑 완료: %d개 항목 -> %d개 그룹 (그룹 크기 %s)",
            len(masked),
            len(groups_idx),
            [len(g) for g in groups_idx],
        )

        # ③ 그룹별 분석. 맛집은 장소별 결과를 다시 여러 MemoGroup으로 펼친다. (+ 단계별 로그)
        total = len(groups_idx)
        result: list[MemoGroup] = []
        for g_i, idx_list in enumerate(groups_idx, start=1):
            merged = _merge_with_source_refs(masked, idx_list)
            merged_mapping: dict[str, str] = {}
            for i in idx_list:
                merged_mapping.update(mask_results[i].mapping)

            logger.info("③ 그룹 %d/%d 분석 시작: 멤버 %d개, chars=%d", g_i, total, len(idx_list), len(merged))
            g_started = time.perf_counter()
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
            member_ids = [items[i].clientId for i in idx_list]
            result.append(MemoGroup(memberClientIds=member_ids, analysis=analysis))
            logger.info(
                "③ 그룹 %d/%d 분석 완료: category=%s isUseful=%s (%.0fms)",
                g_i,
                total,
                analysis.category,
                analysis.isUseful,
                (time.perf_counter() - g_started) * 1000,
            )

        logger.info(
            "배치 분석 완료: groups=%d (%.0fms)", len(result), (time.perf_counter() - started) * 1000
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
            logger.info("임베딩 클라이언트 없음 -> 그룹핑 생략, 항목별 개별 처리")
            return [[i] for i in range(len(masked))]  # 그룹핑 없이 개별 처리(안전한 기본값)

        # 임베딩(그룹핑)용으로만 전처리(노이즈 제거). LLM 분석엔 원본 masked를 쓴다.
        # 전처리로 전부 비면 원본으로 폴백(빈 문자열 임베딩 방지).
        embed_inputs = [(preprocess_for_embedding(m) or m)[:_EMBED_MAX_CHARS] for m in masked]
        total_chars = sum(len(t) for t in embed_inputs)
        logger.info("임베딩 호출: %d건 (전처리 후 총 chars=%d)", len(embed_inputs), total_chars)
        embed_started = time.perf_counter()
        try:
            embeddings = self._embedding.embed(embed_inputs)
        except EmbeddingError as exc:
            logger.warning(
                "embedding 실패, 그룹핑 생략(개별 처리): %s (%.0fms)",
                type(exc).__name__,
                (time.perf_counter() - embed_started) * 1000,
            )
            return [[i] for i in range(len(masked))]

        # 임베딩 수 != 항목 수면 그룹핑이 항목을 누락시킬 수 있으므로, 안전하게 개별 처리.
        if len(embeddings) != len(masked):
            logger.warning(
                "embedding 수 불일치(%d != %d), 그룹핑 생략(개별 처리)",
                len(embeddings),
                len(masked),
            )
            return [[i] for i in range(len(masked))]

        dim = len(embeddings[0]) if embeddings else 0
        logger.info(
            "임베딩 완료: %d벡터 dim=%d (%.0fms)",
            len(embeddings),
            dim,
            (time.perf_counter() - embed_started) * 1000,
        )
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
    buckets: dict[str, dict[str, list]] = {}

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
            if key not in {"sourceRefs", "title", "topicKey", "summary", "recommendedAction"}
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
        topic_key = entry.get("topicKey")
        normalized_topic = (
            _normalize_topic_key(topic_key)
            if isinstance(topic_key, str) and topic_key.strip()
            else None
        )
        source_key = ",".join(sorted(refs))
        bucket_key = f"topic:{normalized_topic}" if normalized_topic else f"sources:{source_key}"
        bucket = buckets.setdefault(
            bucket_key,
            {"entries": [], "member_ids": [], "topic_keys": [], "source_keys": []},
        )
        bucket["entries"].append(
            {
                "title": str(title)[:40],
                "summary": str(summary),
                "recommendedAction": recommended_action,
                "details": place_details,
            }
        )
        bucket["member_ids"].extend(member_ids)
        if normalized_topic:
            bucket["topic_keys"].append(normalized_topic)
        bucket["source_keys"].append(source_key)

    # 누락된 원본이 있으면 사진을 임의의 가게에 붙이지 않고 개별 재분석한다.
    if covered_refs != valid_refs:
        return None

    groups: list[MemoGroup] = []
    for bucket in buckets.values():
        entries = bucket["entries"]
        member_ids = list(dict.fromkeys(bucket["member_ids"]))
        if len(entries) == 1:
            entry = entries[0]
            details = entry["details"]
            title = entry["title"]
            summary = entry["summary"]
            recommended_action = entry["recommendedAction"]
            server_memo_id = _build_restaurant_server_memo_id(details, title)
        else:
            details = _combine_comparison_details(entries)
            topic_keys = bucket["topic_keys"]
            title = topic_keys[0] if topic_keys else analysis.title
            summary = " ".join(
                dict.fromkeys(entry["summary"] for entry in entries if entry["summary"])
            )
            recommended_action = analysis.recommendedAction
            topic_key = topic_keys[0] if topic_keys else None
            server_memo_id = _build_comparison_server_memo_id(topic_key, bucket["source_keys"])

        groups.append(
            MemoGroup(
                memberClientIds=member_ids,
                analysis=analysis.model_copy(
                    update={
                        "serverMemoId": server_memo_id,
                        "title": str(title)[:40],
                        "summary": str(summary),
                        "recommendedAction": recommended_action,
                        "details": details,
                    }
                ),
            )
        )
    return groups


def _normalize_topic_key(value: str) -> str:
    return " ".join(value.casefold().split()).strip()


def _build_restaurant_server_memo_id(details: dict, title: str) -> str | None:
    restaurant = details.get("restaurant")
    if not isinstance(restaurant, dict):
        return None
    place_id = restaurant.get("mapProviderPlaceId")
    if isinstance(place_id, str) and place_id.strip():
        return f"kakao:{place_id.strip()}"
    name = restaurant.get("name") or title
    address = restaurant.get("address") or restaurant.get("roadAddress")
    cleaned_name = _normalize_topic_key(name) if isinstance(name, str) else None
    cleaned_address = _normalize_topic_key(address) if isinstance(address, str) else None
    if cleaned_name and cleaned_address:
        return f"restaurant:{cleaned_name}:{cleaned_address}"
    return None


def _build_comparison_server_memo_id(topic_key: str | None, source_keys: list[str]) -> str | None:
    if topic_key:
        return f"comparison:{topic_key}"
    if source_keys:
        return f"comparison:{_normalize_topic_key(source_keys[0])}"
    return None


def _combine_comparison_details(entries: list[dict]) -> dict:
    menus: list[dict] = []
    tags: list[str] = []
    features: list[str] = []
    actions: list[dict] = []
    confidences: list[float] = []

    for entry in entries:
        details = entry["details"]
        restaurant = details.get("restaurant")
        if not isinstance(restaurant, dict):
            continue
        restaurant_name = restaurant.get("name")
        for menu in restaurant.get("menus") or []:
            if not isinstance(menu, dict):
                continue
            combined_menu = dict(menu)
            menu_name = str(combined_menu.get("name") or "메뉴")
            if restaurant_name:
                combined_menu["name"] = f"{restaurant_name} · {menu_name}"
            menus.append(combined_menu)
        tags.extend(str(value) for value in restaurant.get("tags") or [] if value)
        features.extend(str(value) for value in restaurant.get("features") or [] if value)
        actions.extend(
            value
            for value in restaurant.get("recommendedActions") or []
            if isinstance(value, dict)
        )
        confidence = details.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    return {
        "restaurant": {
            "name": None,
            "address": None,
            "roadAddress": None,
            "neighborhood": None,
            "latitude": None,
            "longitude": None,
            "mapProvider": None,
            "mapProviderPlaceId": None,
            "menus": menus,
            "estimatedPricePerPersonMin": None,
            "estimatedPricePerPersonMax": None,
            "tags": list(dict.fromkeys(tags))[:5],
            "features": list(dict.fromkeys(features))[:5],
            "recommendedActions": actions[:3],
        },
        "group": None,
        "confidence": min(confidences, default=0.5),
        "needsUserReview": True,
    }
