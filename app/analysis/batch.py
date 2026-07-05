import logging

from app.analysis.grouping import group_indices
from app.analysis.service import AnalysisService
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
            return []

        # ① 서버측 2차 마스킹 (클라이언트가 이미 마스킹했어도 한 번 더)
        masked = [mask_text(item.maskedText) for item in items]
        timestamps = [item.capturedAt for item in items]

        # ② 유사 항목 그룹핑 (임베딩 불가 시 각 항목을 개별 그룹으로)
        groups_idx = self._group(masked, timestamps)

        # ③ 그룹별로 텍스트를 합쳐 LLM 분석 -> 그룹당 메모 1개
        result: list[MemoGroup] = []
        for idx_list in groups_idx:
            merged = "\n\n".join(masked[i] for i in idx_list if masked[i])
            analysis = self._analysis.analyze(merged, locale)
            member_ids = [items[i].clientId for i in idx_list]
            result.append(MemoGroup(memberClientIds=member_ids, analysis=analysis))
        return result

    def _group(self, masked: list[str], timestamps: list[int | None]) -> list[list[int]]:
        if self._embedding is None:
            return [[i] for i in range(len(masked))]  # 그룹핑 없이 개별 처리(안전한 기본값)

        try:
            embeddings = self._embedding.embed([m[:_EMBED_MAX_CHARS] for m in masked])
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
