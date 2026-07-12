import logging

from app.analysis.grouping import group_indices
from app.analysis.matching import centroid, match_groups_to_memos
from app.analysis.preprocess import preprocess_for_embedding
from app.analysis.service import AnalysisService, restore_in_response
from app.config import Settings, settings
from app.embeddings.base import EmbeddingClient, EmbeddingError
from app.embeddings.factory import build_embedding_client
from app.models import AnalyzeBatchItem, ExistingMemo, MemoGroup
from app.privacy import mask_text

logger = logging.getLogger("capturemate.batch")

# 임베딩 입력이 너무 길면 토큰 한계에 걸릴 수 있어 앞부분만 사용(그룹핑 판단엔 충분).
_EMBED_MAX_CHARS = 2000


class BatchAnalyzer:
    """하루치 스크린샷을 배치로 받아 유사한 것끼리 묶고, 그룹당 하나의 메모를 만든다.

    흐름: 서버측 마스킹 -> 임베딩 -> (cosine × 시간가중) 배치내 그룹핑
          -> (신규) 각 그룹을 기존 메모와 cosine 비교해 합병 여부 판단
          -> 합병이면 '업데이트', 아니면 '새 분석' -> 그룹별 메모.
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

    def analyze_batch(
        self,
        items: list[AnalyzeBatchItem],
        locale: str,
        existing_memos: list[ExistingMemo] | None = None,
    ) -> list[MemoGroup]:
        if not items:
            return []
        existing_memos = (existing_memos or [])[: self._config.merge_max_candidates]

        # ① 서버측 마스킹 (+ 복원 매핑). namespace로 항목 간 토큰 충돌 방지([PHONE_0_1] 등)
        mask_results = [mask_text(item.maskedText, namespace=f"{i}_") for i, item in enumerate(items)]
        masked = [mr.text for mr in mask_results]
        timestamps = [item.capturedAt for item in items]

        # ①-b 후보 메모도 서버측에서 마스킹(OCR 텍스트와 동일 취급). title+summary를 한 덩어리로.
        memo_mask = [
            mask_text(f"{m.title}\n{m.summary}", namespace=f"m{k}_")
            for k, m in enumerate(existing_memos)
        ]
        memo_masked = [mm.text for mm in memo_mask]

        # ② 임베딩(새 항목 + 후보 메모를 한 번에). 임베딩 불가 시 (None, None).
        item_vecs, memo_vecs = self._embed(masked, memo_masked)

        # ③ 배치 내부 그룹핑(시간 가중 O). 임베딩 없으면 각 항목을 개별 그룹으로.
        groups_idx = self._group(masked, timestamps, item_vecs)

        # ④ 세션 간 매칭(시간 가중 X). 그룹별 합병 대상 메모 인덱스(없으면 None).
        merge_targets = self._match(groups_idx, item_vecs, memo_vecs)

        # ⑤ 그룹별로 합병(업데이트) 또는 새 분석 -> 그룹당 메모 1개 (+ 소프트 토큰 복원)
        result: list[MemoGroup] = []
        for g, idx_list in enumerate(groups_idx):
            merged = "\n\n".join(masked[i] for i in idx_list if masked[i])
            merged_mapping: dict[str, str] = {}
            for i in idx_list:
                merged_mapping.update(mask_results[i].mapping)
            member_ids = [items[i].clientId for i in idx_list]

            target = merge_targets[g] if merge_targets is not None else None
            if target is not None:
                memo = existing_memos[target]
                # 업데이트 응답엔 후보 메모의 소프트 토큰도 등장할 수 있어 복원 매핑에 합친다.
                merged_mapping.update(memo_mask[target].mapping)
                analysis = self._analysis.update(memo, memo_masked[target], merged, locale)
                analysis = restore_in_response(analysis, merged_mapping)
                result.append(
                    MemoGroup(
                        memberClientIds=member_ids,
                        analysis=analysis,
                        mergeTargetMemoId=memo.memoId,
                    )
                )
            else:
                analysis = self._analysis.analyze(merged, locale)
                analysis = restore_in_response(analysis, merged_mapping)
                result.append(MemoGroup(memberClientIds=member_ids, analysis=analysis))
        return result

    def _embed(
        self, item_texts: list[str], memo_texts: list[str]
    ) -> tuple[list[list[float]] | None, list[list[float]] | None]:
        """새 항목 + 후보 메모를 '한 번의 호출'로 임베딩한다.

        반환: (item_vecs, memo_vecs). 임베딩 불가/실패/개수 불일치면 (None, None)로
        폴백해 그룹핑·매칭을 안전하게 생략한다(전체 요청은 절대 깨지지 않음).
        """
        if self._embedding is None:
            return None, None

        # 그룹핑용으로만 전처리(노이즈 제거). LLM 분석엔 원본 masked를 쓴다.
        # 후보 메모는 이미 짧은 title/summary라 전처리 없이 그대로 사용.
        item_inputs = [(preprocess_for_embedding(m) or m)[:_EMBED_MAX_CHARS] for m in item_texts]
        memo_inputs = [m[:_EMBED_MAX_CHARS] for m in memo_texts]
        all_inputs = item_inputs + memo_inputs
        if not all_inputs:
            return None, None

        try:
            vecs = self._embedding.embed(all_inputs)
        except EmbeddingError as exc:
            logger.warning("embedding 실패, 그룹핑/매칭 생략: %s", type(exc).__name__)
            return None, None

        if len(vecs) != len(all_inputs):
            logger.warning("embedding 수 불일치(%d != %d), 그룹핑/매칭 생략", len(vecs), len(all_inputs))
            return None, None

        split = len(item_inputs)
        return vecs[:split], vecs[split:]

    def _group(
        self,
        masked: list[str],
        timestamps: list[int | None],
        item_vecs: list[list[float]] | None,
    ) -> list[list[int]]:
        if item_vecs is None:
            return [[i] for i in range(len(masked))]  # 그룹핑 없이 개별 처리(안전한 기본값)

        return group_indices(
            item_vecs,
            timestamps,
            threshold=self._config.grouping_threshold,
            tau_seconds=self._config.grouping_tau_seconds,
            w_min=self._config.grouping_w_min,
        )

    def _match(
        self,
        groups_idx: list[list[int]],
        item_vecs: list[list[float]] | None,
        memo_vecs: list[list[float]] | None,
    ) -> list[int | None] | None:
        # 임베딩이 없거나 후보 메모가 없으면 합병 없음 -> 전부 새 메모.
        if item_vecs is None or not memo_vecs:
            return None

        group_vecs = [centroid([item_vecs[i] for i in idx_list]) for idx_list in groups_idx]
        return match_groups_to_memos(group_vecs, memo_vecs, threshold=self._config.merge_threshold)
