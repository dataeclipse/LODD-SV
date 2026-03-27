
from __future__ import annotations

import abc
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeBase(abc.ABC):

    @abc.abstractmethod
    def query(self, query_text: str, **kwargs: Any) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_fact(self, key: str) -> Optional[str]:
        pass


class SQLKnowledgeBase(KnowledgeBase):

    def __init__(
        self,
        db_path: str = ":memory:",
        create_schema: bool = True,
    ) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        if create_schema:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    text TEXT
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON facts(subject)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_text ON facts(text)")
            self.conn.commit()

    def add_fact(self, subject: str, predicate: str, obj: str, text: Optional[str] = None) -> None:
        text = text or f"{subject} {predicate} {obj}"
        self.conn.execute(
            "INSERT INTO facts (subject, predicate, object, text) VALUES (?, ?, ?, ?)",
            (subject, predicate, obj, text),
        )
        self.conn.commit()

    def query(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT id, subject, predicate, object, text FROM facts
            WHERE text LIKE ? OR subject LIKE ? OR object LIKE ?
            LIMIT ?
            """,
            (f"%{query_text}%", f"%{query_text}%", f"%{query_text}%", limit),
        )
        rows = cur.fetchall()
        return [
            {"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3], "text": r[4]}
            for r in rows
        ]

    def get_fact(self, key: str) -> Optional[str]:
        try:
            kid = int(key)
            cur = self.conn.execute("SELECT text FROM facts WHERE id = ?", (kid,))
        except ValueError:
            cur = self.conn.execute(
                "SELECT text FROM facts WHERE subject = ? OR object = ? LIMIT 1",
                (key, key),
            )
        row = cur.fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self.conn.close()


class InMemoryKnowledgeBase(KnowledgeBase):

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._documents: List[Dict[str, Any]] = []

    def add(self, key: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._store[key] = text
        self._documents.append({"key": key, "text": text, **(meta or {})})

    def query(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = (query_text or "").lower().strip()
        if not q:
            return []


        stop = {"the", "of", "and", "a", "an", "to", "in", "on", "for", "is", "are", "was", "were"}
        q_tokens = [t for t in re.findall(r"[a-z0-9]+", q) if t not in stop]


        if len(q_tokens) <= 1:
            out = []
            for doc in self._documents:
                if q in (doc.get("text", "") or "").lower() or q in (doc.get("key", "") or "").lower():
                    out.append(doc)
                    if len(out) >= limit:
                        break
            return out

        scored: List[tuple[int, int, Dict[str, Any]]] = []
        for idx, doc in enumerate(self._documents):
            text = (doc.get("text", "") or "").lower()
            doc_tokens = set(re.findall(r"[a-z0-9]+", text))
            overlap = sum(1 for t in q_tokens if t in doc_tokens)
            if overlap > 0:
                scored.append((overlap, -idx, doc))

        scored.sort(reverse=True)
        return [d for (_, _, d) in scored[:limit]]

    def get_fact(self, key: str) -> Optional[str]:
        return self._store.get(key)
