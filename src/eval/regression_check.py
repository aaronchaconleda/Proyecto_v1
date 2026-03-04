from __future__ import annotations

from typing import Dict


def compare_runs(current: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    for key, value in current.items():
        delta[key] = value - baseline.get(key, 0.0)
    return delta
