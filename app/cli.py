from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Optional

import requests
import typer

from src.config.logging_setup import setup_logging
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


def _model_id(model: dict) -> str:
    return str(model.get("id") or model.get("name") or "")


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


def _parse_doc_id_filter(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    parts = [item.strip() for item in raw.split(",")]
    return [item for item in parts if item]


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def _bootstrap():
    settings = load_settings()
    logger = setup_logging(settings.data_dir / "logs")
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
    return settings, sqlite_store, sync, lmstudio, vector, hybrid, logger


def _load_models_or_fail(lmstudio: LMStudioClient):
    try:
        return lmstudio.list_models()
    except requests.RequestException as exc:
        raise typer.BadParameter(
            "No se pudo conectar con LM Studio para listar modelos. Verifica que este activo."
        ) from exc


def _validate_doc_filter_or_fail(sqlite_store: SQLiteStore, doc_ids: list[str]) -> None:
    if not doc_ids:
        return
    existing = set(sqlite_store.existing_doc_ids(doc_ids))
    missing = [doc_id for doc_id in doc_ids if doc_id not in existing]
    if missing:
        raise typer.BadParameter(f"doc_id_filter invalido. No existen: {', '.join(missing)}")


@cli.command("init-session")
def init_session(session_id: Optional[str] = typer.Option(None, "--session-id")):
    _, sqlite_store, _, _, _, _, logger = _bootstrap()
    sid = sqlite_store.create_session(session_id=session_id)
    logger.info("session_created session_id=%s", sid)
    typer.echo(f"session_id={sid}")


@cli.command("list-docs")
def list_docs_cmd():
    _, sqlite_store, _, _, _, _, logger = _bootstrap()
    docs = sqlite_store.list_documents_summary()
    if not docs:
        typer.echo("No hay documentos indexados.")
        sqlite_store.close()
        return

    typer.echo("Documentos indexados:")
    for idx, item in enumerate(docs, start=1):
        typer.echo(
            f"{idx}. doc_id={item['doc_id']} chunks={item['chunk_count']} lang={item.get('language') or '-'} "
            f"embedding={item['embedding_model']}"
        )
        typer.echo(f"   path={item['path']}")
        typer.echo(f"   created_at={item['created_at']}")
    logger.info("list_docs total=%s", len(docs))
    sqlite_store.close()


@cli.command("delete-doc")
def delete_doc_cmd(
    doc_id: str = typer.Option(..., "--doc-id", help="Identificador del documento a borrar."),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Muestra lo que se borraria sin ejecutar cambios.",
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Confirma borrado real (solo aplica con --no-dry-run).",
    ),
):
    settings, sqlite_store, _, _, _, _, logger = _bootstrap()
    summary = sqlite_store.get_document_summary(doc_id)
    if not summary:
        typer.echo(f"No existe doc_id={doc_id}")
        sqlite_store.close()
        return

    typer.echo("Documento objetivo:")
    typer.echo(
        f"doc_id={summary['doc_id']} chunks={summary['chunk_count']} embedding={summary['embedding_model']}"
    )
    typer.echo(f"path={summary['path']}")
    typer.echo(f"created_at={summary['created_at']}")

    if dry_run:
        typer.echo("Dry-run activo: no se realizaron cambios.")
        sqlite_store.close()
        return

    if not confirm:
        typer.echo("Operacion cancelada. Usa --confirm para ejecutar borrado real.")
        sqlite_store.close()
        return

    chroma_store = ChromaStore(settings.chroma_dir)
    try:
        chroma_store.delete_by_doc_id(doc_id)
        deleted_chunks = sqlite_store.delete_document(doc_id)
        logger.info("delete_doc doc_id=%s chunks=%s", doc_id, deleted_chunks)
        typer.echo(f"Borrado completado: doc_id={doc_id} chunks_eliminados={deleted_chunks}")
    except Exception as exc:
        logger.exception("delete_doc_error doc_id=%s", doc_id)
        typer.echo(f"Error al borrar documento: {exc}")
    sqlite_store.close()


@cli.command("vacuum-db")
def vacuum_db_cmd():
    settings = load_settings()
    db_path = settings.sqlite_path
    before = db_path.stat().st_size if db_path.exists() else 0

    _, sqlite_store, _, _, _, _, logger = _bootstrap()
    sqlite_store.vacuum()
    sqlite_store.close()

    after = db_path.stat().st_size if db_path.exists() else 0
    typer.echo(
        f"VACUUM rag.db completado: antes={_format_size(before)} despues={_format_size(after)} "
        f"liberado={_format_size(max(0, before - after))}"
    )
    logger.info("vacuum_db before=%s after=%s", before, after)


@cli.command("vacuum-chroma")
def vacuum_chroma_cmd(
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Confirma compactacion de chroma.sqlite3. Recomendado ejecutarlo sin indexado/chat activo.",
    ),
):
    logger = setup_logging(load_settings().data_dir / "logs")
    if not confirm:
        typer.echo("Operacion cancelada. Usa --confirm para compactar Chroma.")
        return

    settings = load_settings()
    chroma_sqlite = settings.chroma_dir / "chroma.sqlite3"
    if not chroma_sqlite.exists():
        typer.echo(f"No existe archivo de Chroma: {chroma_sqlite}")
        return

    before = chroma_sqlite.stat().st_size
    try:
        conn = sqlite3.connect(str(chroma_sqlite))
        conn.execute("VACUUM")
        conn.close()
    except sqlite3.OperationalError as exc:
        logger.warning("vacuum_chroma_locked path=%s error=%s", chroma_sqlite, exc)
        typer.echo(f"No se pudo compactar Chroma (posible bloqueo): {exc}")
        typer.echo("Cierra procesos de chat/indexado y vuelve a intentarlo.")
        return

    after = chroma_sqlite.stat().st_size
    typer.echo(
        f"VACUUM chroma.sqlite3 completado: antes={_format_size(before)} despues={_format_size(after)} "
        f"liberado={_format_size(max(0, before - after))}"
    )
    logger.info("vacuum_chroma before=%s after=%s path=%s", before, after, chroma_sqlite)


@cli.command("index")
def index_cmd(
    doc_path: str = typer.Argument(..., help="Ruta del documento PDF/TXT/MD."),
    doc_id: Optional[str] = typer.Option(None, "--doc-id"),
    lang: str = typer.Option("es", "--lang"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model"),
):
    settings, sqlite_store, sync, lmstudio, _, _, logger = _bootstrap()
    path = Path(doc_path).resolve()
    if not path.exists():
        raise typer.BadParameter(f"No existe archivo: {path}")

    selected_embedding_model = embedding_model
    models = _load_models_or_fail(lmstudio)
    if selected_embedding_model:
        available = {_model_id(m) for m in models}
        if selected_embedding_model not in available:
            sqlite_store.close()
            raise typer.BadParameter(f"Embedding model no disponible en LM Studio: {selected_embedding_model}")
    else:
        _, embedding_models = _split_chat_and_embedding_models(models)
        selected_embedding_model = _pick_model_interactive(
            embedding_models,
            title="Modelos de embedding disponibles:",
            default_name=settings.embedding_model,
        )
    settings.embedding_model = selected_embedding_model

    final_doc_id = doc_id or path.stem
    try:
        count = index_document(
            settings=settings,
            sync_service=sync,
            client=lmstudio,
            doc_id=final_doc_id,
            file_path=path,
            language=lang,
        )
        logger.info("index_ok doc_id=%s chunks=%s embedding=%s", final_doc_id, count, selected_embedding_model)
        typer.echo(f"doc_id={final_doc_id} chunks_indexados={count}")
    except requests.RequestException:
        logger.exception("index_lmstudio_error doc_id=%s", final_doc_id)
        typer.echo("Error conectando con LM Studio durante indexado. Verifica que este activo y modelo cargado.")
    except Exception as exc:
        logger.exception("index_error doc_id=%s", final_doc_id)
        typer.echo(f"Error en indexado: {exc}")
    sqlite_store.close()


@cli.command("chat")
def chat_cmd(
    session_id: str = typer.Option(..., "--session-id"),
    question: str = typer.Argument(...),
    chat_model: Optional[str] = typer.Option(None, "--chat-model"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    doc_id_filter: Optional[str] = typer.Option(
        None,
        "--doc-id-filter",
        help="Filtra retrieval por doc_id. Multiples por coma: doc1,doc2",
    ),
):
    settings, sqlite_store, _, lmstudio, vector, hybrid, logger = _bootstrap()
    sqlite_store.create_session(session_id=session_id)
    if sqlite_store.count_chunks() <= 0:
        sqlite_store.close()
        raise typer.BadParameter("No hay documentos indexados. Ejecuta index o wizard primero.")

    parsed_filter = _parse_doc_id_filter(doc_id_filter)
    _validate_doc_filter_or_fail(sqlite_store, parsed_filter)

    selected_chat_model = chat_model
    models = _load_models_or_fail(lmstudio)
    if selected_chat_model:
        available = {_model_id(m) for m in models}
        if selected_chat_model not in available:
            sqlite_store.close()
            raise typer.BadParameter(f"Chat model no disponible en LM Studio: {selected_chat_model}")
    else:
        chat_models, _ = _split_chat_and_embedding_models(models)
        selected_chat_model = _pick_model_interactive(
            chat_models,
            title="Modelos conversacionales disponibles:",
            default_name=chat_model or settings.chat_model,
        )

    try:
        result = answer_question(
            settings=settings,
            sqlite_store=sqlite_store,
            lmstudio_client=lmstudio,
            vector_retriever=vector,
            hybrid_retriever=hybrid if settings.retrieval_hybrid else None,
            session_id=session_id,
            question=question,
            chat_model=selected_chat_model,
            top_k=top_k,
            doc_id_filter=parsed_filter,
        )
    except requests.RequestException:
        logger.exception("chat_lmstudio_error session_id=%s", session_id)
        typer.echo("Error conectando con LM Studio durante chat. Verifica servidor y modelo.")
        sqlite_store.close()
        return
    except Exception as exc:
        logger.exception("chat_error session_id=%s", session_id)
        typer.echo(f"Error en chat: {exc}")
        sqlite_store.close()
        return

    typer.echo("Respuesta:")
    typer.echo(result["answer"])
    typer.echo("\nFuentes:")
    for idx, chunk in enumerate(result["chunks"], start=1):
        typer.echo(
            f"{idx}. [{chunk['doc_id']}:p{chunk.get('page', '?')}:{chunk['chunk_id']}] "
            f"score={chunk.get('score_final', chunk.get('score_vector', 0.0)):.4f}"
        )
    typer.echo(f"\nlatency_ms={result['latency_ms']} query_id={result['query_id']}")
    logger.info("chat_ok session_id=%s top_k=%s chunks=%s", session_id, top_k or settings.retrieval_top_k, len(result["chunks"]))
    sqlite_store.close()


@cli.command("wizard")
def wizard_cmd(
    no_index: bool = typer.Option(
        False,
        "--no-index",
        help="No indexar documento nuevo; usar solo la base RAG ya existente.",
    ),
):
    settings, sqlite_store, sync, lmstudio, vector, hybrid, logger = _bootstrap()

    typer.echo("Conectando con LM Studio para obtener modelos...")
    models = _load_models_or_fail(lmstudio)
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

    session_id = typer.prompt("Session ID", default="demo-es")
    top_k = int(typer.prompt("Top-K retrieval", default=str(settings.retrieval_top_k)))
    filter_raw = typer.prompt("Filtro doc_id (opcional, separados por coma)", default="")
    doc_id_filter = _parse_doc_id_filter(filter_raw)
    should_index = False if no_index else typer.confirm("Quieres indexar un documento nuevo ahora?", default=True)

    settings.chat_model = selected_chat
    settings.embedding_model = selected_embedding
    sqlite_store.create_session(session_id=session_id)

    if should_index:
        doc_input = typer.prompt("Ruta del documento (PDF/TXT/MD)")
        doc_path = Path(doc_input).expanduser().resolve()
        if not doc_path.exists():
            raise typer.BadParameter(f"No existe archivo: {doc_path}")
        doc_id = typer.prompt("Doc ID", default=doc_path.stem)
        lang = typer.prompt("Idioma", default="es")

        typer.echo("\nIndexando documento...")
        try:
            chunk_count = index_document(
                settings=settings,
                sync_service=sync,
                client=lmstudio,
                doc_id=doc_id,
                file_path=doc_path,
                language=lang,
            )
            logger.info("wizard_index_ok doc_id=%s chunks=%s", doc_id, chunk_count)
            typer.echo(f"Indexado completado. chunks={chunk_count}")
        except requests.RequestException:
            logger.exception("wizard_index_lmstudio_error doc_id=%s", doc_id)
            sqlite_store.close()
            raise typer.BadParameter("Error conectando con LM Studio durante indexado.") from None
        except Exception as exc:
            logger.exception("wizard_index_error doc_id=%s", doc_id)
            sqlite_store.close()
            raise typer.BadParameter(f"Error durante indexado: {exc}") from None
    else:
        total_chunks = sqlite_store.count_chunks()
        if total_chunks <= 0:
            sqlite_store.close()
            raise typer.BadParameter(
                "No hay conocimiento indexado todavia. Indexa al menos un documento o ejecuta wizard sin --no-index."
            )
        typer.echo(f"Usando base RAG existente. chunks_disponibles={total_chunks}")

    _validate_doc_filter_or_fail(sqlite_store, doc_id_filter)
    if doc_id_filter:
        typer.echo(f"Filtro activo por doc_id: {', '.join(doc_id_filter)}")
    typer.echo("Modo conversacion listo. Escribe 'salir' para terminar.\n")

    while True:
        question = typer.prompt("Tu pregunta").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            typer.echo("Sesion finalizada.")
            break
        if not question:
            continue

        try:
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
                doc_id_filter=doc_id_filter,
            )
        except requests.RequestException:
            logger.exception("wizard_chat_lmstudio_error session_id=%s", session_id)
            typer.echo("Error conectando con LM Studio durante chat.")
            continue
        except Exception as exc:
            logger.exception("wizard_chat_error session_id=%s", session_id)
            typer.echo(f"Error en chat: {exc}")
            continue
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

