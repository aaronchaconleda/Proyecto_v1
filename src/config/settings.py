from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class AppSettings:
    rag_profile: str
    project_root: Path
    data_dir: Path
    sqlite_path: Path
    chroma_dir: Path
    docs_dir: Path
    llm_provider: str
    lmstudio_base_url: str
    lmstudio_api_key: str
    openai_base_url: str
    openai_api_key: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap_ratio: float
    retrieval_top_k: int
    retrieval_hybrid: bool
    retrieval_vector_weight: float
    retrieval_keyword_weight: float
    history_window_messages: int

    @property
    def chunk_overlap(self) -> int:
        return int(self.chunk_size * self.chunk_overlap_ratio)


def load_settings() -> AppSettings:
    load_dotenv(Path.cwd() / ".env")
    root = Path(os.getenv("RAG_PROJECT_ROOT", Path.cwd())).resolve()
    load_dotenv(root / ".env", override=False)
    raw_profile = os.getenv("RAG_PROFILE", "local").strip().lower()
    rag_profile = raw_profile if raw_profile in {"local", "openai"} else "local"
    data_dir = Path(os.getenv("RAG_DATA_DIR", root / "data")).resolve()
    sqlite_default = data_dir / f"rag_{rag_profile}.db"
    chroma_default = data_dir / f"chroma_{rag_profile}"
    sqlite_path = Path(os.getenv("RAG_SQLITE_PATH", sqlite_default)).resolve()
    chroma_dir = Path(os.getenv("RAG_CHROMA_DIR", chroma_default)).resolve()
    docs_dir = Path(os.getenv("RAG_DOCS_DIR", root / "docs")).resolve()
    default_provider = "openai" if rag_profile == "openai" else "lmstudio"
    default_chat_model = "gpt-5-nano" if rag_profile == "openai" else "qwen/qwen3-4b-thinking-2507"
    default_embedding_model = "text-embedding-3-small" if rag_profile == "openai" else "text-embedding-nomic-embed-code"

    return AppSettings(
        rag_profile=rag_profile,
        project_root=root,
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        chroma_dir=chroma_dir,
        docs_dir=docs_dir,
        llm_provider=os.getenv("RAG_LLM_PROVIDER", default_provider).lower(),
        lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        lmstudio_api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("RAG_CHAT_MODEL", default_chat_model),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", default_embedding_model),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
        chunk_overlap_ratio=float(os.getenv("RAG_CHUNK_OVERLAP_RATIO", "0.15")),
        retrieval_top_k=int(os.getenv("RAG_TOP_K", "6")),
        retrieval_hybrid=os.getenv("RAG_HYBRID", "true").lower() in {"1", "true", "yes"},
        retrieval_vector_weight=float(os.getenv("RAG_VECTOR_WEIGHT", "0.7")),
        retrieval_keyword_weight=float(os.getenv("RAG_KEYWORD_WEIGHT", "0.3")),
        history_window_messages=int(os.getenv("RAG_HISTORY_WINDOW_MESSAGES", "10")),
    )
