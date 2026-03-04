from __future__ import annotations

from typing import List

from src.chunking.chunk_metadata import ChunkRecord
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_store import SQLiteStore


class SyncService:
    def __init__(self, sqlite_store: SQLiteStore, chroma_store: ChromaStore) -> None:
        self.sqlite_store = sqlite_store
        self.chroma_store = chroma_store

    def upsert_chunks(self, chunks: List[ChunkRecord], embeddings: List[List[float]]) -> None:
        self.sqlite_store.upsert_chunks(chunks)
        self.chroma_store.upsert_chunks(chunks, embeddings)
