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


def _pick_model_interactive(models, title: str, default_name: str) -> str:
    typer.echo(f"\n{title}")
    for idx, model in enumerate(models, start=1):
        model_id = model.get("id") or model.get("name") or f"model_{idx}"
        typer.echo(f"{idx}. {model_id}")

    default_idx = 1
    for idx, model in enumerate(models, start=1):
        model_id = (model.get("id") or model.get("name") or "").lower()
        if default_name.lower() in model_id:
            default_idx = idx
            break

    selected = typer.prompt("Selecciona numero", default=str(default_idx))
    try:
        pos = int(selected)
        if pos < 1 or pos > len(models):
            raise ValueError
    except ValueError:
        raise typer.BadParameter("Seleccion invalida de modelo.")

    chosen = models[pos - 1]
    return chosen.get("id") or chosen.get("name")


def _split_chat_and_embedding_models(models):
    chat_models = []
    embedding_models = []
    for item in models:
        model_id = (item.get("id") or item.get("name") or "").lower()
        if "embed" in model_id or "embedding" in model_id:
            embedding_models.append(item)
        else:
            chat_models.append(item)
    if not chat_models:
        chat_models = models
    if not embedding_models:
        embedding_models = models
    return chat_models, embedding_models


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


@cli.command("wizard")
def wizard_cmd():
    settings, sqlite_store, sync, lmstudio, vector, hybrid = _bootstrap()

    typer.echo("Conectando con LM Studio para obtener modelos...")
    models = lmstudio.list_models()
    if not models:
        raise typer.BadParameter("LM Studio no devolvio modelos.")

    chat_models, embedding_models = _split_chat_and_embedding_models(models)

    selected_chat = _pick_model_interactive(
        chat_models,
        title="Modelos conversacionales disponibles:",
        default_name=settings.chat_model,
    )
    selected_embedding = _pick_model_interactive(
        embedding_models,
        title="Modelos de embedding disponibles:",
        default_name=settings.embedding_model,
    )

    doc_input = typer.prompt("Ruta del documento (PDF/TXT/MD)")
    doc_path = Path(doc_input).expanduser().resolve()
    if not doc_path.exists():
        raise typer.BadParameter(f"No existe archivo: {doc_path}")

    session_id = typer.prompt("Session ID", default="demo-es")
    doc_id = typer.prompt("Doc ID", default=doc_path.stem)
    lang = typer.prompt("Idioma", default="es")
    top_k = int(typer.prompt("Top-K retrieval", default=str(settings.retrieval_top_k)))

    settings.chat_model = selected_chat
    settings.embedding_model = selected_embedding
    sqlite_store.create_session(session_id=session_id)

    typer.echo("\nIndexando documento...")
    chunk_count = index_document(
        settings=settings,
        sync_service=sync,
        client=lmstudio,
        doc_id=doc_id,
        file_path=doc_path,
        language=lang,
    )
    typer.echo(f"Indexado completado. chunks={chunk_count}")
    typer.echo("Modo conversacion listo. Escribe 'salir' para terminar.\n")

    while True:
        question = typer.prompt("Tu pregunta").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            typer.echo("Sesion finalizada.")
            break
        if not question:
            continue

        result = answer_question(
            settings=settings,
            sqlite_store=sqlite_store,
            lmstudio_client=lmstudio,
            vector_retriever=vector,
            hybrid_retriever=hybrid if settings.retrieval_hybrid else None,
            session_id=session_id,
            question=question,
            chat_model=selected_chat,
            top_k=top_k,
        )
        typer.echo("\nRespuesta:")
        typer.echo(result["answer"])
        typer.echo("\nFuentes:")
        for idx, chunk in enumerate(result["chunks"], start=1):
            typer.echo(
                f"{idx}. [{chunk['doc_id']}:p{chunk.get('page', '?')}:{chunk['chunk_id']}] "
                f"score={chunk.get('score_final', chunk.get('score_vector', 0.0)):.4f}"
            )
        typer.echo("")

    sqlite_store.close()


if __name__ == "__main__":
    cli()
