from __future__ import annotations

from typing import Dict, List, Optional

from src.storage.chroma_store import ChromaStore


class VectorRetriever:
    def __init__(self, chroma_store: ChromaStore) -> None:
        self.chroma_store = chroma_store

    def retrieve(
        self,
        query_embedding: List[float],
        top_k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        return self.chroma_store.query(query_embedding=query_embedding, top_k=top_k, doc_ids=doc_ids)
