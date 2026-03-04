from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    page: int
    section: str
    offset_start: int
    offset_end: int
    text: str


def make_chunk_id(doc_id: str, page: int, offset_start: int, offset_end: int) -> str:
    raw = f"{doc_id}:{page}:{offset_start}:{offset_end}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
