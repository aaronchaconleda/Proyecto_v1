from __future__ import annotations

from typing import Dict, List

from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.vector_retriever import VectorRetriever


def _rank_score(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 - float(rank - 1) / float(total - 1)


class HybridRetriever:
    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        vector_weight: float,
        keyword_weight: float,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        *,
        query: str,
        query_embedding: List[float],
        top_k: int,
        doc_ids: List[str] | None = None,
    ) -> List[Dict]:
        vector_results = self.vector_retriever.retrieve(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            doc_ids=doc_ids,
        )
        raw_keyword_results = self.keyword_retriever.retrieve(query=query, top_k=top_k * 2, doc_ids=doc_ids)
        keyword_results: List[Dict] = []
        total_kw = len(raw_keyword_results)
        for rank, item in enumerate(raw_keyword_results, start=1):
            keyword_results.append(
                {
                    **item,
                    "score_keyword_raw": float(item.get("score_keyword", 0.0)),
                    "score_keyword": _rank_score(rank, total_kw),
                }
            )

        merged: Dict[str, Dict] = {}
        for item in vector_results:
            merged[item["chunk_id"]] = {**item, "score_keyword": 0.0, "score_keyword_raw": 0.0}
        for item in keyword_results:
            if item["chunk_id"] in merged:
                merged[item["chunk_id"]]["score_keyword"] = item["score_keyword"]
                merged[item["chunk_id"]]["score_keyword_raw"] = item.get("score_keyword_raw", 0.0)
            else:
                merged[item["chunk_id"]] = {**item, "score_vector": 0.0}

        for item in merged.values():
            score_vector = float(item.get("score_vector", 0.0))
            score_keyword = float(item.get("score_keyword", 0.0))
            item["score_final"] = self.vector_weight * score_vector + self.keyword_weight * score_keyword

        ranked = sorted(merged.values(), key=lambda x: x["score_final"], reverse=True)
        return ranked[:top_k]
