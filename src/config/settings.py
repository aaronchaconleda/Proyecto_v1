from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppSettings:
    project_root: Path
    data_dir: Path
    sqlite_path: Path
    chroma_dir: Path
    docs_dir: Path
    lmstudio_base_url: str
    lmstudio_api_key: str
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
    root = Path(os.getenv("RAG_PROJECT_ROOT", Path.cwd())).resolve()
    data_dir = Path(os.getenv("RAG_DATA_DIR", root / "data")).resolve()
    sqlite_path = Path(os.getenv("RAG_SQLITE_PATH", data_dir / "rag.db")).resolve()
    chroma_dir = Path(os.getenv("RAG_CHROMA_DIR", data_dir / "chroma")).resolve()
    docs_dir = Path(os.getenv("RAG_DOCS_DIR", root / "docs")).resolve()

    return AppSettings(
        project_root=root,
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        chroma_dir=chroma_dir,
        docs_dir=docs_dir,
        lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        lmstudio_api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        chat_model=os.getenv("RAG_CHAT_MODEL", "qwen/qwen3-4b-thinking-2507"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-nomic-embed-code"),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
        chunk_overlap_ratio=float(os.getenv("RAG_CHUNK_OVERLAP_RATIO", "0.15")),
        retrieval_top_k=int(os.getenv("RAG_TOP_K", "6")),
        retrieval_hybrid=os.getenv("RAG_HYBRID", "true").lower() in {"1", "true", "yes"},
        retrieval_vector_weight=float(os.getenv("RAG_VECTOR_WEIGHT", "0.7")),
        retrieval_keyword_weight=float(os.getenv("RAG_KEYWORD_WEIGHT", "0.3")),
        history_window_messages=int(os.getenv("RAG_HISTORY_WINDOW_MESSAGES", "10")),
    )
