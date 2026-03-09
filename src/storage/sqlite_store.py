from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.chunking.chunk_metadata import ChunkRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_fts_query(query: str) -> str:
    # Keep only word-like tokens and quote each term to avoid FTS syntax errors.
    tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens)


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                language TEXT,
                embedding_model TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page INTEGER NOT NULL,
                section TEXT,
                offset_start INTEGER NOT NULL,
                offset_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(chunk_id, text, content='chunks', content_rowid='rowid');

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (new.rowid, new.chunk_id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text) VALUES('delete', old.rowid, old.chunk_id, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, text) VALUES('delete', old.rowid, old.chunk_id, old.text);
                INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (new.rowid, new.chunk_id, new.text);
            END;

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                model TEXT NOT NULL,
                content TEXT NOT NULL,
                retrieval_meta TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS queries (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_message_id TEXT,
                assistant_message_id TEXT,
                latency_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS retrieval_logs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                page INTEGER,
                rank INTEGER NOT NULL,
                score_vector REAL,
                score_keyword REAL,
                score_final REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def upsert_document(self, doc_id: str, path: str, language: str, embedding_model: str) -> None:
        self.conn.execute(
            """
            INSERT INTO documents(doc_id, path, language, embedding_model, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                path=excluded.path,
                language=excluded.language,
                embedding_model=excluded.embedding_model
            """,
            (doc_id, path, language, embedding_model, _utc_now()),
        )
        self.conn.commit()

    def upsert_chunks(self, chunks: Iterable[ChunkRecord]) -> None:
        rows = []
        now = _utc_now()
        for chunk in chunks:
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.page,
                    chunk.section,
                    chunk.offset_start,
                    chunk.offset_end,
                    chunk.text,
                    json.dumps(asdict(chunk), ensure_ascii=True),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO chunks(chunk_id, doc_id, page, section, offset_start, offset_end, text, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                page=excluded.page,
                section=excluded.section,
                offset_start=excluded.offset_start,
                offset_end=excluded.offset_end,
                text=excluded.text,
                metadata_json=excluded.metadata_json
            """,
            rows,
        )
        self.conn.commit()

    def create_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, user_id, created_at) VALUES (?, ?, ?)",
            (session_id, user_id, _utc_now()),
        )
        self.conn.commit()
        return session_id

    def add_message(
        self,
        *,
        session_id: str,
        role: str,
        model: str,
        content: str,
        retrieval_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        message_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO messages(id, session_id, role, model, content, retrieval_meta, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                role,
                model,
                content,
                json.dumps(retrieval_meta, ensure_ascii=True) if retrieval_meta else None,
                _utc_now(),
            ),
        )
        self.conn.commit()
        return message_id

    def get_recent_messages(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, role, model, content, retrieval_meta, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def search_chunks_fts(
        self,
        query: str,
        top_k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        safe_query = _safe_fts_query(query)
        if not safe_query:
            return []
        try:
            if doc_ids:
                placeholders = ", ".join(["?"] * len(doc_ids))
                sql = f"""
                    SELECT c.chunk_id, c.doc_id, c.page, c.section, c.text, bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    AND c.doc_id IN ({placeholders})
                    ORDER BY score ASC
                    LIMIT ?
                """
                params = [safe_query, *doc_ids, top_k]
                rows = self.conn.execute(sql, params).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT c.chunk_id, c.doc_id, c.page, c.section, c.text, bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY score ASC
                    LIMIT ?
                    """,
                    (safe_query, top_k),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25 in SQLite returns lower is better; invert to align with vector score.
        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "page": row["page"],
                "section": row["section"],
                "text": row["text"],
                "score_keyword": 1.0 / (1.0 + float(row["score"])),
            }
            for row in rows
        ]

    def create_query_log(
        self,
        *,
        session_id: str,
        user_message_id: str,
        assistant_message_id: str,
        latency_ms: int,
        usage: Optional[Dict[str, Any]],
    ) -> str:
        query_id = str(uuid.uuid4())
        usage = usage or {}
        self.conn.execute(
            """
            INSERT INTO queries(
                id, session_id, user_message_id, assistant_message_id,
                latency_ms, prompt_tokens, completion_tokens, total_tokens, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                query_id,
                session_id,
                user_message_id,
                assistant_message_id,
                latency_ms,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                usage.get("total_tokens"),
                _utc_now(),
            ),
        )
        self.conn.commit()
        return query_id

    def add_retrieval_logs(
        self,
        *,
        session_id: str,
        query_id: str,
        results: List[Dict[str, Any]],
    ) -> None:
        rows = []
        now = _utc_now()
        for rank, item in enumerate(results, start=1):
            rows.append(
                (
                    str(uuid.uuid4()),
                    session_id,
                    query_id,
                    item["chunk_id"],
                    item["doc_id"],
                    item.get("page"),
                    rank,
                    item.get("score_vector"),
                    item.get("score_keyword"),
                    item.get("score_final"),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO retrieval_logs(
                id, session_id, query_id, chunk_id, doc_id, page, rank,
                score_vector, score_keyword, score_final, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def count_chunks(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
        return int(row["total"]) if row else 0

    def list_documents_summary(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                d.doc_id,
                d.path,
                d.language,
                d.embedding_model,
                d.created_at,
                COUNT(c.chunk_id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.doc_id
            GROUP BY d.doc_id, d.path, d.language, d.embedding_model, d.created_at
            ORDER BY d.created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """
            SELECT
                d.doc_id,
                d.path,
                d.language,
                d.embedding_model,
                d.created_at,
                COUNT(c.chunk_id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.doc_id
            WHERE d.doc_id = ?
            GROUP BY d.doc_id, d.path, d.language, d.embedding_model, d.created_at
            """,
            (doc_id,),
        ).fetchone()
        return dict(row) if row else None

    def delete_document(self, doc_id: str) -> int:
        chunk_row = self.conn.execute(
            "SELECT COUNT(*) AS total FROM chunks WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        chunk_count = int(chunk_row["total"]) if chunk_row else 0
        cur = self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        if cur.rowcount <= 0:
            return 0
        return chunk_count

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")

    def close(self) -> None:
        self.conn.close()
