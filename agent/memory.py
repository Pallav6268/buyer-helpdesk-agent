"""Long-term memory: short facts about each buyer, kept across chats in a small SQLite file.

The agent saves facts with the ``remember`` tool and sees them at the start of every turn.
Buyers can see and delete everything stored about them from the sidebar.
"""

from __future__ import annotations

import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_PATH = Path(os.getenv("MEMORY_DB", Path(__file__).resolve().parent.parent / "memory.db"))
MAX_FACTS = 30  # per buyer; the oldest are dropped first
MAX_FACT_CHARS = 200

# Never store identifiers like these, even if a buyer types them.
SENSITIVE = [
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "a PAN number"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "an Aadhaar or card number"),
    (re.compile(r"\b\d{9,18}\b"), "an account or phone number"),
]


def sensitive_reason(fact: str) -> str | None:
    for pattern, label in SENSITIVE:
        if pattern.search(fact.upper()):
            return label
    return None


class MemoryStore:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = str(path)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS facts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, buyer_id TEXT NOT NULL, "
                "fact TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success and is always closed."""
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, buyer_id: str, fact: str) -> str:
        fact = " ".join(fact.split())[:MAX_FACT_CHARS]
        if not fact:
            return "Nothing to remember."
        if reason := sensitive_reason(fact):
            return f"Not saved: it looks like {reason}, which this assistant never stores."
        with self._db() as db:
            existing = [row[0].lower() for row in db.execute("SELECT fact FROM facts WHERE buyer_id = ?", (buyer_id,))]
            if fact.lower() in existing:
                return "Already remembered."
            db.execute(
                "INSERT INTO facts (buyer_id, fact, created_at) VALUES (?, ?, ?)",
                (buyer_id, fact, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            db.execute(
                "DELETE FROM facts WHERE buyer_id = ? AND id NOT IN "
                "(SELECT id FROM facts WHERE buyer_id = ? ORDER BY id DESC LIMIT ?)",
                (buyer_id, buyer_id, MAX_FACTS),
            )
        return f"Remembered: {fact}"

    def list(self, buyer_id: str) -> list[dict]:
        with self._db() as db:
            rows = db.execute(
                "SELECT id, fact, created_at FROM facts WHERE buyer_id = ? ORDER BY id", (buyer_id,)
            ).fetchall()
        return [{"id": r[0], "fact": r[1], "created_at": r[2]} for r in rows]

    def delete(self, buyer_id: str, fact_id: int) -> None:
        with self._db() as db:
            db.execute("DELETE FROM facts WHERE buyer_id = ? AND id = ?", (buyer_id, fact_id))

    def clear(self, buyer_id: str) -> None:
        with self._db() as db:
            db.execute("DELETE FROM facts WHERE buyer_id = ?", (buyer_id,))
