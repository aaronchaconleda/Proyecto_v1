from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict


def save_eval_dataset(path: Path, items: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=True, indent=2), encoding="utf-8")
