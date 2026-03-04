from __future__ import annotations

from typing import Dict, List


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    return len(top.intersection(rel)) / len(rel)


def mean_reciprocal_rank(ranked_ids: List[str], relevant_ids: List[str]) -> float:
    relevant = set(relevant_ids)
    for idx, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / idx
    return 0.0


def summarize_metrics(values: List[Dict[str, float]]) -> Dict[str, float]:
    if not values:
        return {}
    keys = values[0].keys()
    return {k: sum(v[k] for v in values) / len(values) for k in keys}
