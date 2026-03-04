from __future__ import annotations

from typing import List

from src.chunking.chunk_metadata import ChunkRecord, make_chunk_id


def chunk_text(
    *,
    doc_id: str,
    page: int,
    text: str,
    chunk_size: int,
    overlap: int,
    section: str = "default",
) -> List[ChunkRecord]:
    if not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    chunks: List[ChunkRecord] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_text_value = " ".join(chunk_words).strip()
        if chunk_text_value:
            chunk_id = make_chunk_id(doc_id, page, start, end)
            chunks.append(
                ChunkRecord(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    page=page,
                    section=section,
                    offset_start=start,
                    offset_end=end,
                    text=chunk_text_value,
                )
            )
        if end >= len(words):
            break
        start += step
    return chunks
