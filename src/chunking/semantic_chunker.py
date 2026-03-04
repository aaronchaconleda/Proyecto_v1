from __future__ import annotations

from typing import List

from src.chunking.chunk_metadata import ChunkRecord
from src.chunking.window_chunker import chunk_text


def semantic_chunk_by_paragraph(
    *,
    doc_id: str,
    page: int,
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[ChunkRecord]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    records: List[ChunkRecord] = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        section = f"p{idx}"
        records.extend(
            chunk_text(
                doc_id=doc_id,
                page=page,
                text=paragraph,
                chunk_size=chunk_size,
                overlap=overlap,
                section=section,
            )
        )
    return records
