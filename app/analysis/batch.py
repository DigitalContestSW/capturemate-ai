import logging
import time

from app.analysis.grouping import group_indices
from app.analysis.preprocess import preprocess_for_embedding
from app.analysis.service import AnalysisService, restore_in_response
from app.config import Settings, settings
from app.embeddings.base import EmbeddingClient, EmbeddingError
from app.embeddings.factory import build_embedding_client
from app.models import AnalyzeBatchItem, MemoGroup
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

        # ③ 그룹별로 텍스트를 합쳐 LLM 분석 -> 그룹당 메모 1개 (+ 소프트 토큰 복원)
        total = len(groups_idx)
        result: list[MemoGroup] = []
        for g_i, idx_list in enumerate(groups_idx, start=1):
            merged = "\n\n".join(masked[i] for i in idx_list if masked[i])
            merged_mapping: dict[str, str] = {}
            for i in idx_list:
                merged_mapping.update(mask_results[i].mapping)

            logger.info("③ 그룹 %d/%d 분석 시작: 멤버 %d개, chars=%d", g_i, total, len(idx_list), len(merged))
            g_started = time.perf_counter()
            analysis = self._analysis.analyze(merged, locale)
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
