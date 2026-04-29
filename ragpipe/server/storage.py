"""SQLite persistence for conversations and messages.

Lightweight, zero-deps store for chat history. Schema:
    conversations(id, title, model, provider, created_at, updated_at)
    messages(id, conversation_id, role, content, sources_json, model, tokens_used, latency_ms, created_at)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


class ConversationStore:
    """SQLite-backed conversation history. Thread-safe via per-call connections."""

    def __init__(self, db_path: str = "ragpipe.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT,
                    provider TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    model TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(updated_at DESC);
            """)

    # --- conversations ---

    def create_conversation(
        self,
        title: str = "New chat",
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> dict[str, Any]:
        cid = str(uuid.uuid4())
        now = _now_ms()
        with self._conn() as c:
            c.execute(
                "INSERT INTO conversations (id, title, model, provider, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cid, title, model, provider, now, now),
            )
        return {
            "id": cid,
            "title": title,
            "model": model,
            "provider": provider,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def list_conversations(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT c.*, "
                "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count "
                "FROM conversations c ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, cid: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
            if not row:
                return None
            messages = c.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (cid,),
            ).fetchall()
        result = dict(row)
        result["messages"] = [self._format_message(m) for m in messages]
        return result

    def update_conversation_title(self, cid: str, title: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now_ms(), cid),
            )
            return cur.rowcount > 0

    def delete_conversation(self, cid: str) -> bool:
        with self._conn() as c:
            c.execute("PRAGMA foreign_keys = ON")
            cur = c.execute("DELETE FROM conversations WHERE id = ?", (cid,))
            return cur.rowcount > 0

    # --- messages ---

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        mid = str(uuid.uuid4())
        now = _now_ms()
        sources_json = json.dumps(sources) if sources else None
        with self._conn() as c:
            c.execute(
                "INSERT INTO messages (id, conversation_id, role, content, sources_json, "
                "model, tokens_used, latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mid, conversation_id, role, content, sources_json, model,
                 tokens_used, latency_ms, now),
            )
            c.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        return {
            "id": mid,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "model": model,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "created_at": now,
        }

    @staticmethod
    def _format_message(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        sj = d.pop("sources_json", None)
        d["sources"] = json.loads(sj) if sj else []
        return d
