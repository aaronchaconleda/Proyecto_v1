from __future__ import annotations

import time
from typing import Dict, List, Optional

from src.config.settings import AppSettings
from src.llm.answer_generator import generate_answer
from src.llm.lmstudio_client import LMStudioClient
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import rerank_candidates
from src.retrieval.vector_retriever import VectorRetriever
from src.storage.sqlite_store import SQLiteStore


def _conversation_for_llm(rows: List[Dict]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for row in rows:
        role = row["role"]
        if role in {"user", "assistant", "system"}:
            messages.append({"role": role, "content": row["content"]})
    return messages


def answer_question(
    *,
    settings: AppSettings,
    sqlite_store: SQLiteStore,
    lmstudio_client: LMStudioClient,
    vector_retriever: VectorRetriever,
    hybrid_retriever: Optional[HybridRetriever],
    session_id: str,
    question: str,
    chat_model: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict:
    top_k = top_k or settings.retrieval_top_k
    chat_model = chat_model or settings.chat_model

    user_message_id = sqlite_store.add_message(
        session_id=session_id,
        role="user",
        model=chat_model,
        content=question,
    )

    start = time.perf_counter()
    query_embedding = lmstudio_client.embed_texts(settings.embedding_model, [question])[0]
    if settings.retrieval_hybrid and hybrid_retriever is not None:
        candidates = hybrid_retriever.retrieve(query=question, query_embedding=query_embedding, top_k=top_k)
    else:
        candidates = vector_retriever.retrieve(query_embedding=query_embedding, top_k=top_k)
        for item in candidates:
            item["score_final"] = item.get("score_vector", 0.0)

    retrieved = rerank_candidates(candidates, top_k=top_k)
    history_rows = sqlite_store.get_recent_messages(session_id, settings.history_window_messages)
    history_rows = [row for row in history_rows if row["id"] != user_message_id]
    history_for_llm = _conversation_for_llm(history_rows)

    answer, usage = generate_answer(
        client=lmstudio_client,
        chat_model=chat_model,
        question=question,
        retrieved_chunks=retrieved,
        conversation_history=history_for_llm,
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    assistant_message_id = sqlite_store.add_message(
        session_id=session_id,
        role="assistant",
        model=chat_model,
        content=answer,
        retrieval_meta={"top_k": top_k, "chunks": [item["chunk_id"] for item in retrieved]},
    )

    query_id = sqlite_store.create_query_log(
        session_id=session_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        latency_ms=elapsed_ms,
        usage=usage,
    )
    sqlite_store.add_retrieval_logs(session_id=session_id, query_id=query_id, results=retrieved)

    return {
        "answer": answer,
        "chunks": retrieved,
        "usage": usage,
        "latency_ms": elapsed_ms,
        "query_id": query_id,
    }
