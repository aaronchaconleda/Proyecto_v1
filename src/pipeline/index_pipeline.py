from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from src.chunking.chunk_metadata import ChunkRecord
from src.chunking.semantic_chunker import semantic_chunk_by_paragraph
from src.config.settings import AppSettings
from src.ingestion.cleaning import basic_clean, remove_repeated_lines
from src.ingestion.normalizer import normalize_text
from src.ingestion.pdf_loader import load_pdf_pages
from src.llm.lmstudio_client import LMStudioClient
from src.storage.sync_service import SyncService


def _read_text_document(path: Path) -> List[Tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    return [(1, text)]


def load_document_pages(path: Path) -> List[Tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_pages(path)
    if suffix in {".txt", ".md"}:
        return _read_text_document(path)
    raise ValueError(f"Formato no soportado: {suffix}. Usa PDF/TXT/MD.")


def index_document(
    *,
    settings: AppSettings,
    sync_service: SyncService,
    client: LMStudioClient,
    doc_id: str,
    file_path: Path,
    language: str = "es",
) -> int:
    pages = load_document_pages(file_path)
    raw_texts = [text for _, text in pages]
    cleaned_pages = remove_repeated_lines(raw_texts)

    all_chunks: List[ChunkRecord] = []
    for (page_number, _), cleaned in zip(pages, cleaned_pages):
        normalized = normalize_text(basic_clean(cleaned))
        chunks = semantic_chunk_by_paragraph(
            doc_id=doc_id,
            page=page_number,
            text=normalized,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    embeddings = client.embed_texts(
        model=settings.embedding_model,
        texts=[chunk.text for chunk in all_chunks],
    )
    sync_service.sqlite_store.upsert_document(
        doc_id=doc_id,
        path=str(file_path),
        language=language,
        embedding_model=settings.embedding_model,
    )
    sync_service.upsert_chunks(all_chunks, embeddings)
    return len(all_chunks)
