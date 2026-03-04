from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.config.settings import load_settings
from src.llm.lmstudio_client import LMStudioClient
from src.pipeline.index_pipeline import index_document
from src.pipeline.qa_pipeline import answer_question
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_store import SQLiteStore
from src.storage.sync_service import SyncService

cli = typer.Typer(help="RAG local con LM Studio + Chroma + SQLite")


def _bootstrap():
    settings = load_settings()
    sqlite_store = SQLiteStore(settings.sqlite_path)
    chroma_store = ChromaStore(settings.chroma_dir)
    sync = SyncService(sqlite_store, chroma_store)
    lmstudio = LMStudioClient(settings.lmstudio_base_url, settings.lmstudio_api_key)
    vector = VectorRetriever(chroma_store)
    keyword = KeywordRetriever(sqlite_store)
    hybrid = HybridRetriever(
        vector_retriever=vector,
        keyword_retriever=keyword,
        vector_weight=settings.retrieval_vector_weight,
        keyword_weight=settings.retrieval_keyword_weight,
    )
    return settings, sqlite_store, sync, lmstudio, vector, hybrid


@cli.command("init-session")
def init_session(session_id: Optional[str] = typer.Option(None, "--session-id")):
    _, sqlite_store, _, _, _, _ = _bootstrap()
    sid = sqlite_store.create_session(session_id=session_id)
    typer.echo(f"session_id={sid}")


@cli.command("index")
def index_cmd(
    doc_path: str = typer.Argument(..., help="Ruta del documento PDF/TXT/MD."),
    doc_id: Optional[str] = typer.Option(None, "--doc-id"),
    lang: str = typer.Option("es", "--lang"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model"),
):
    settings, sqlite_store, sync, lmstudio, _, _ = _bootstrap()
    path = Path(doc_path).resolve()
    if not path.exists():
        raise typer.BadParameter(f"No existe archivo: {path}")

    if embedding_model:
        settings.embedding_model = embedding_model

    final_doc_id = doc_id or path.stem
    count = index_document(
        settings=settings,
        sync_service=sync,
        client=lmstudio,
        doc_id=final_doc_id,
        file_path=path,
        language=lang,
    )
    typer.echo(f"doc_id={final_doc_id} chunks_indexados={count}")
    sqlite_store.close()


@cli.command("chat")
def chat_cmd(
    session_id: str = typer.Option(..., "--session-id"),
    question: str = typer.Argument(...),
    chat_model: Optional[str] = typer.Option(None, "--chat-model"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
):
    settings, sqlite_store, _, lmstudio, vector, hybrid = _bootstrap()
    sqlite_store.create_session(session_id=session_id)

    result = answer_question(
        settings=settings,
        sqlite_store=sqlite_store,
        lmstudio_client=lmstudio,
        vector_retriever=vector,
        hybrid_retriever=hybrid if settings.retrieval_hybrid else None,
        session_id=session_id,
        question=question,
        chat_model=chat_model,
        top_k=top_k,
    )

    typer.echo("Respuesta:")
    typer.echo(result["answer"])
    typer.echo("\nFuentes:")
    for idx, chunk in enumerate(result["chunks"], start=1):
        typer.echo(
            f"{idx}. [{chunk['doc_id']}:p{chunk.get('page', '?')}:{chunk['chunk_id']}] "
            f"score={chunk.get('score_final', chunk.get('score_vector', 0.0)):.4f}"
        )
    typer.echo(f"\nlatency_ms={result['latency_ms']} query_id={result['query_id']}")
    sqlite_store.close()


if __name__ == "__main__":
    cli()
