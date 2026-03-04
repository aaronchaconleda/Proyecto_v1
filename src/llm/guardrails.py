from __future__ import annotations


def enforce_no_hallucination(answer: str) -> str:
    text = answer.strip()
    if not text:
        return "No encontrado en el documento."
    return text
