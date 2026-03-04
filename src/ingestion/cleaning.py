from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List


def remove_repeated_lines(texts: Iterable[str], min_repetitions: int = 3) -> List[str]:
    all_lines = []
    for text in texts:
        all_lines.extend([line.strip() for line in text.splitlines() if line.strip()])
    line_counts = Counter(all_lines)
    noisy_lines = {line for line, count in line_counts.items() if count >= min_repetitions}

    cleaned: List[str] = []
    for text in texts:
        kept = [line for line in text.splitlines() if line.strip() not in noisy_lines]
        cleaned.append("\n".join(kept))
    return cleaned


def basic_clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
