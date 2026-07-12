from typing import Optional

from app.analysis.grouping import _cosine  # 동일 코사인 정의 재사용


def centroid(vectors: list[list[float]]) -> list[float]:
    """벡터들의 평균(그룹 대표 벡터). 코사인은 방향만 보므로 정규화 없이 평균해도 무방."""
    n = len(vectors)
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    return [x / n for x in acc]


def match_groups_to_memos(
    group_vectors: list[list[float]],
    memo_vectors: list[list[float]],
    threshold: float,
) -> list[Optional[int]]:
    """각 새 그룹을 합병할 기존 메모 인덱스로 매핑한다(없으면 None).

    판단식: cosine(group, memo) ≥ threshold  (배치 내부 그룹핑과 달리 '시간 가중 없음').
      - 지난주 메모라도 내용이 맞으면 유효한 합병 대상이므로 시간은 게이트로 쓰지 않는다.
      - threshold는 배치 내부보다 높게(보수적) — 잘못된 합병이 기존 메모를 오염시키는
        비용이 새 메모를 하나 더 만드는 비용보다 크기 때문.

    충돌 처리: 유사도가 높은 쌍부터 그리디로 배정하되,
      - 한 메모는 최대 한 그룹에만 합병(중복 합병 방지),
      - 한 그룹은 최대 한 메모에만 합병.
      이미 배정된 쪽과 겹치면 그 그룹은 합병하지 않고 새 메모로 남긴다(None).
    """
    pairs: list[tuple[float, int, int]] = []
    for gi, gv in enumerate(group_vectors):
        for mi, mv in enumerate(memo_vectors):
            sim = _cosine(gv, mv)
            if sim >= threshold:
                pairs.append((sim, gi, mi))

    pairs.sort(key=lambda p: p[0], reverse=True)  # 유사도 높은 순

    assigned: dict[int, int] = {}
    used_memos: set[int] = set()
    for _sim, gi, mi in pairs:
        if gi in assigned or mi in used_memos:
            continue
        assigned[gi] = mi
        used_memos.add(mi)

    return [assigned.get(gi) for gi in range(len(group_vectors))]
