from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import chromadb

from src.chunking.chunk_metadata import ChunkRecord


class ChromaStore:
    def __init__(self, chroma_dir: Path, collection_name: str = "rag_chunks") -> None:
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: Iterable[ChunkRecord], embeddings: List[List[float]]) -> None:
        chunk_list = list(chunks)
        self.collection.upsert(
            ids=[c.chunk_id for c in chunk_list],
            documents=[c.text for c in chunk_list],
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "section": c.section,
                    "offset_start": c.offset_start,
                    "offset_end": c.offset_end,
                }
                for c in chunk_list
            ],
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        where = None
        if doc_ids:
            where = {"doc_id": {"$in": doc_ids}} if len(doc_ids) > 1 else {"doc_id": doc_ids[0]}

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        items = []
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist)
            items.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id"),
                    "page": meta.get("page"),
                    "section": meta.get("section"),
                    "text": doc,
                    "score_vector": score,
                }
            )
        return items
