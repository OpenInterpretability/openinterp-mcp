"""Tiny sqlite-backed vector store. No external vector DB.

Schema:
  entries(slug TEXT PRIMARY KEY, title TEXT, author TEXT, type TEXT, model_id TEXT,
          claim TEXT, hf_url TEXT, doi TEXT, manifest_sha256 TEXT,
          created_at TEXT, embedding BLOB)

Embeddings are stored as float32 arrays in BLOB. Cosine similarity computed in numpy in-process.
At 10k entries × 1536d float32 = 60 MB. Fits easily; brute-force fast enough.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_DB_PATH = Path("~/.openinterp/atlas.db").expanduser()


@dataclass
class AtlasEntry:
    slug: str
    title: str
    author: str
    type: str
    model_id: Optional[str]
    claim: Optional[str]
    hf_url: Optional[str]
    doi: Optional[str]
    manifest_sha256: str
    created_at: str
    embedding: np.ndarray


class VectorStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                slug TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                type TEXT NOT NULL,
                model_id TEXT,
                claim TEXT,
                hf_url TEXT,
                doi TEXT,
                manifest_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        self.conn.commit()

    def upsert(self, entry: AtlasEntry) -> None:
        emb_blob = entry.embedding.astype(np.float32).tobytes()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO entries
            (slug, title, author, type, model_id, claim, hf_url, doi, manifest_sha256, created_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (entry.slug, entry.title, entry.author, entry.type, entry.model_id, entry.claim,
             entry.hf_url, entry.doi, entry.manifest_sha256, entry.created_at, emb_blob),
        )
        self.conn.commit()

    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               filter_type: Optional[str] = None) -> List[Tuple[float, AtlasEntry]]:
        rows = list(self._iter_rows(filter_type))
        if not rows:
            return []
        embs = np.stack([np.frombuffer(r[10], dtype=np.float32) for r in rows])
        q = query_embedding.astype(np.float32)
        q /= np.linalg.norm(q) + 1e-8
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        sims = embs_norm @ q
        idx = np.argsort(sims)[::-1][:top_k]
        return [(float(sims[i]), self._row_to_entry(rows[i])) for i in idx]

    def _iter_rows(self, filter_type: Optional[str]) -> Iterable[tuple]:
        cur = self.conn.cursor()
        if filter_type:
            cur.execute("SELECT * FROM entries WHERE type = ?", (filter_type,))
        else:
            cur.execute("SELECT * FROM entries")
        return cur.fetchall()

    @staticmethod
    def _row_to_entry(row: tuple) -> AtlasEntry:
        return AtlasEntry(
            slug=row[0], title=row[1], author=row[2], type=row[3], model_id=row[4],
            claim=row[5], hf_url=row[6], doi=row[7], manifest_sha256=row[8], created_at=row[9],
            embedding=np.frombuffer(row[10], dtype=np.float32),
        )

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
