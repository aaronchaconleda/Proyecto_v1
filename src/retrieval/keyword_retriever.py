from __future__ import annotations

from typing import Dict, List

from src.storage.sqlite_store import SQLiteStore


class KeywordRetriever:
    def __init__(self, sqlite_store: SQLiteStore) -> None:
        self.sqlite_store = sqlite_store

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        return self.sqlite_store.search_chunks_fts(query=query, top_k=top_k)
