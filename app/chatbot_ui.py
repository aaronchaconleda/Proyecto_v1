from __future__ import annotations

from pathlib import Path
import sys
import uuid
from typing import Any, Optional

import chainlit as cl
import requests

# Ensure project root is available when Chainlit loads this module directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.logging_setup import setup_logging
from src.config.settings import AppSettings, load_settings
from src.llm.client_factory import create_llm_client
from src.pipeline.qa_pipeline import answer_question
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.storage.chroma_store import ChromaStore
from src.storage.sqlite_store import SQLiteStore


class ChatRuntime:
    def __init__(
        self,
        *,
        settings: AppSettings,
        sqlite_store: SQLiteStore,
        llm_client: Any,
        vector_retriever: VectorRetriever,
        hybrid_retriever: HybridRetriever,
        session_id: str,
        chat_model: str,
        profile: str,
        logger: Any,
    ) -> None:
        self.settings = settings
        self.sqlite_store = sqlite_store
        self.llm_client = llm_client
        self.vector_retriever = vector_retriever
        self.hybrid_retriever = hybrid_retriever
        self.session_id = session_id
        self.chat_model = chat_model
        self.profile = profile
        self.logger = logger


def _model_id(model: dict) -> str:
    return str(model.get("id") or model.get("name") or "")


def _split_chat_and_embedding_models(models):
    chat_models = []
    embedding_models = []
    for item in models:
        model_id = _model_id(item).lower()
        if "embed" in model_id or "embedding" in model_id:
            embedding_models.append(item)
        else:
            chat_models.append(item)
    if not chat_models:
        chat_models = models
    if not embedding_models:
        embedding_models = models
    return chat_models, embedding_models


def _existing_embedding_models(sqlite_store: SQLiteStore) -> list[str]:
    docs = sqlite_store.list_documents_summary()
    return sorted({str(item.get("embedding_model")) for item in docs if item.get("embedding_model")})


def _select_profile(raw: Optional[str], default_profile: str) -> str:
    value = (raw or "").strip().lower()
    if value in {"local", "openai"}:
        return value
    return default_profile


def _build_profile_settings(profile: str) -> AppSettings:
    settings = load_settings()
    settings.rag_profile = profile
    settings.sqlite_path = (settings.data_dir / f"rag_{profile}.db").resolve()
    settings.chroma_dir = (settings.data_dir / f"chroma_{profile}").resolve()
    settings.llm_provider = "openai" if profile == "openai" else "lmstudio"
    return settings


def _pick_chat_model_or_default(llm_client: Any, default_chat_model: str) -> str:
    models = llm_client.list_models()
    if not models:
        return default_chat_model
    chat_models, _ = _split_chat_and_embedding_models(models)
    if not chat_models:
        return default_chat_model
    available = {_model_id(m) for m in chat_models}
    if default_chat_model in available:
        return default_chat_model
    return _model_id(chat_models[0]) or default_chat_model


def _resolve_chat_model_choice(chat_models, default_chat_model: str, raw_choice: Optional[str]) -> str:
    ids = [_model_id(item) for item in chat_models if _model_id(item)]
    if not ids:
        return default_chat_model

    default_model = default_chat_model if default_chat_model in ids else ids[0]
    raw = (raw_choice or "").strip()
    if not raw:
        return default_model
    if raw.isdigit():
        pos = int(raw)
        if 1 <= pos <= len(ids):
            return ids[pos - 1]
        return default_model
    if raw in ids:
        return raw
    return default_model


def _bootstrap_runtime(profile: str) -> ChatRuntime:
    settings = _build_profile_settings(profile)
    logger = setup_logging(settings.data_dir / "logs")
    sqlite_store = SQLiteStore(settings.sqlite_path)
    chroma_store = ChromaStore(settings.chroma_dir)
    llm_client = create_llm_client(settings)
    vector = VectorRetriever(chroma_store)
    keyword = KeywordRetriever(sqlite_store)
    hybrid = HybridRetriever(
        vector_retriever=vector,
        keyword_retriever=keyword,
        vector_weight=settings.retrieval_vector_weight,
        keyword_weight=settings.retrieval_keyword_weight,
    )

    if sqlite_store.count_chunks() <= 0:
        raise ValueError(
            f"No hay documentos indexados en perfil '{profile}'. "
            "Un admin debe indexar primero desde CLI (python -m app.cli wizard)."
        )

    existing_embeddings = _existing_embedding_models(sqlite_store)
    if len(existing_embeddings) == 1:
        settings.embedding_model = existing_embeddings[0]

    chat_model = _pick_chat_model_or_default(llm_client, settings.chat_model)
    session_id = f"ui-{profile}-{uuid.uuid4().hex[:8]}"
    sqlite_store.create_session(session_id=session_id)

    return ChatRuntime(
        settings=settings,
        sqlite_store=sqlite_store,
        llm_client=llm_client,
        vector_retriever=vector,
        hybrid_retriever=hybrid,
        session_id=session_id,
        chat_model=chat_model,
        profile=profile,
        logger=logger,
    )


@cl.on_chat_start
async def on_chat_start():
    base_settings = load_settings()
    ask = await cl.AskUserMessage(
        content=(
            "Perfil RAG para esta sesion (`local` o `openai`).\n"
            f"Si envias vacio, uso `{base_settings.rag_profile}`."
        ),
        timeout=120,
    ).send()
    profile = _select_profile(ask.get("output") if ask else None, base_settings.rag_profile)

    try:
        runtime = _bootstrap_runtime(profile)
    except Exception as exc:
        await cl.Message(content=f"No se pudo iniciar el chatbot: {exc}").send()
        return

    try:
        models = runtime.llm_client.list_models()
    except Exception:
        models = []
    chat_models, _ = _split_chat_and_embedding_models(models)
    if chat_models:
        ids = [_model_id(item) for item in chat_models if _model_id(item)]
        default_model = runtime.chat_model if runtime.chat_model in ids else (ids[0] if ids else runtime.chat_model)
        default_idx = ids.index(default_model) + 1 if default_model in ids else 1
        lines = [f"{idx}. {model_id}" for idx, model_id in enumerate(ids, start=1)]
        ask_chat_model = await cl.AskUserMessage(
            content=(
                "Modelos conversacionales disponibles:\n"
                + "\n".join(lines)
                + f"\n\nSelecciona numero o escribe id [default: {default_idx}]"
            ),
            timeout=180,
        ).send()
        runtime.chat_model = _resolve_chat_model_choice(
            chat_models=chat_models,
            default_chat_model=runtime.chat_model,
            raw_choice=ask_chat_model.get("output") if ask_chat_model else None,
        )

    cl.user_session.set("runtime", runtime)
    docs = runtime.sqlite_store.count_documents()
    chunks = runtime.sqlite_store.count_chunks()
    await cl.Message(
        content=(
            "Chat listo.\n"
            f"perfil={runtime.profile} session_id={runtime.session_id}\n"
            f"chat_model={runtime.chat_model} embedding_model={runtime.settings.embedding_model}\n"
            f"docs={docs} chunks={chunks}"
        )
    ).send()


@cl.on_chat_end
async def on_chat_end():
    runtime: Optional[ChatRuntime] = cl.user_session.get("runtime")
    if runtime:
        runtime.sqlite_store.close()


@cl.on_message
async def on_message(message: cl.Message):
    runtime: Optional[ChatRuntime] = cl.user_session.get("runtime")
    if not runtime:
        await cl.Message(content="Sesion no inicializada. Recarga la pagina.").send()
        return

    question = (message.content or "").strip()
    if not question:
        return

    if question.lower() in {"/status", "status"}:
        await cl.Message(
            content=(
                f"perfil={runtime.profile}\n"
                f"session_id={runtime.session_id}\n"
                f"chat_model={runtime.chat_model}\n"
                f"embedding_model={runtime.settings.embedding_model}"
            )
        ).send()
        return

    try:
        result = answer_question(
            settings=runtime.settings,
            sqlite_store=runtime.sqlite_store,
            lmstudio_client=runtime.llm_client,
            vector_retriever=runtime.vector_retriever,
            hybrid_retriever=runtime.hybrid_retriever if runtime.settings.retrieval_hybrid else None,
            session_id=runtime.session_id,
            question=question,
            chat_model=runtime.chat_model,
            top_k=runtime.settings.retrieval_top_k,
            doc_id_filter=None,
        )
    except requests.RequestException as exc:
        runtime.logger.exception("ui_chat_provider_error session_id=%s profile=%s", runtime.session_id, runtime.profile)
        await cl.Message(content=f"Error de proveedor LLM: {exc}").send()
        return
    except Exception as exc:
        runtime.logger.exception("ui_chat_error session_id=%s profile=%s", runtime.session_id, runtime.profile)
        await cl.Message(content=f"Error en chat: {exc}").send()
        return

    sources = []
    for idx, chunk in enumerate(result["chunks"], start=1):
        score = chunk.get("score_final", chunk.get("score_vector", 0.0))
        sources.append(f"{idx}. [{chunk['doc_id']}:p{chunk.get('page', '?')}:{chunk['chunk_id']}] score={score:.4f}")

    out = result["answer"]
    if sources:
        out = f"{out}\n\nFuentes:\n" + "\n".join(sources)
    await cl.Message(content=out).send()
