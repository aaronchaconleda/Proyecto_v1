from __future__ import annotations

from typing import Dict, List


def rerank_candidates(candidates: List[Dict], top_k: int) -> List[Dict]:
    ranked = sorted(candidates, key=lambda x: float(x.get("score_final", x.get("score_vector", 0.0))), reverse=True)
    return ranked[:top_k]
