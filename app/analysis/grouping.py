import logging
import math
from typing import Optional

logger = logging.getLogger("capturemate.grouping")


def _cosine(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도(방향 유사도). 0~1 근방, 클수록 의미가 비슷."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _time_weight(dt_seconds: float, tau_seconds: float, w_min: float) -> float:
    """시간 간격에 따른 가중치. Δt=0이면 1, 멀어질수록 감쇠(하한 w_min)."""
    if tau_seconds <= 0:
        return 1.0
    return max(w_min, math.exp(-dt_seconds / tau_seconds))


def group_indices(
    embeddings: list[list[float]],
    timestamps_ms: list[Optional[int]],
    threshold: float,
    tau_seconds: float,
    w_min: float,
) -> list[list[int]]:
    """유사 항목을 같은 그룹으로 묶어 인덱스 리스트들의 리스트를 반환한다.

    판단식: effective = cosine(i, j) × timeWeight(Δt)  →  threshold 이상이면 같은 그룹.
    - 텍스트(cosine)가 게이트: 낮으면 시간이 가까워도 묶이지 않는다(무관한 연속 캡처 방지).
    - 시간은 가중치: 멀어질수록 effective를 낮춰 분리 유도.
    Union-Find(단일 연결) 방식이라 A~B, B~C면 A·B·C가 한 그룹이 된다.
    """
    n = len(embeddings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 경로 압축
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    linked = 0  # threshold를 넘겨 실제로 묶인 쌍의 수(임계값 튜닝 감 잡기용)
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(embeddings[i], embeddings[j])
            if sim <= 0.0:
                continue
            ti, tj = timestamps_ms[i], timestamps_ms[j]
            if ti is None or tj is None:
                time_weight = 1.0  # 시간 정보 없으면 감점하지 않고 텍스트로만 판단
            else:
                time_weight = _time_weight(abs(ti - tj) / 1000.0, tau_seconds, w_min)
            if sim * time_weight >= threshold:
                union(i, j)
                linked += 1

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    result = list(groups.values())
    logger.debug(
        "group_indices: n=%d threshold=%.2f 연결쌍=%d -> 그룹 %d개", n, threshold, linked, len(result)
    )
    return result
