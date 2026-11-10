from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from app.store.db import Database


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex
    return f"{prefix}{raw}" if prefix else raw


class Repository:
    """Persistence for documents, conversations, orders, tickets and feedback."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Documents & chunks
    # ------------------------------------------------------------------
    async def find_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        )
        return dict(row) if row else None

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return dict(row) if row else None

    async def list_documents(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """
            SELECT d.id, d.title, d.filename, d.size_bytes, d.created_at,
                   d.embedding_provider, d.embedding_model, d.embedding_dim,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """
        )
        return [dict(r) for r in rows]

    async def list_documents_for_reindex(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            "SELECT id, title, filename, raw_text FROM documents ORDER BY filename"
        )
        return [dict(r) for r in rows]

    async def insert_document(
        self, doc: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> None:
        async with self.db.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO documents
                    (id, title, filename, content_hash, raw_text, size_bytes,
                     embedding_provider, embedding_model, embedding_dim, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc["id"],
                    doc["title"],
                    doc["filename"],
                    doc["content_hash"],
                    doc["raw_text"],
                    doc["size_bytes"],
                    doc["embedding_provider"],
                    doc["embedding_model"],
                    doc["embedding_dim"],
                    doc["created_at"],
                ),
            )
            await conn.executemany(
                """
                INSERT INTO chunks (id, doc_id, seq, section, content, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c["id"],
                        c["doc_id"],
                        c["seq"],
                        c["section"],
                        c["content"],
                        c["embedding"],
                        c["created_at"],
                    )
                    for c in chunks
                ],
            )

    async def replace_document_chunks(
        self,
        doc_id: str,
        chunks: list[dict[str, Any]],
        provider: str,
        model: str,
        dim: int,
    ) -> None:
        async with self.db.transaction() as conn:
            await conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            await conn.executemany(
                """
                INSERT INTO chunks (id, doc_id, seq, section, content, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c["id"],
                        c["doc_id"],
                        c["seq"],
                        c["section"],
                        c["content"],
                        c["embedding"],
                        c["created_at"],
                    )
                    for c in chunks
                ],
            )
            await conn.execute(
                """
                UPDATE documents
                SET embedding_provider = ?, embedding_model = ?, embedding_dim = ?
                WHERE id = ?
                """,
                (provider, model, dim, doc_id),
            )

    async def delete_document(self, doc_id: str) -> bool:
        changed = await self.db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return changed > 0

    async def list_index_rows(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """
            SELECT c.id, c.doc_id, c.seq, c.section, c.content, c.embedding,
                   d.title, d.filename, d.embedding_provider, d.embedding_model,
                   d.embedding_dim
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            ORDER BY d.created_at, c.seq
            """
        )
        return [dict(r) for r in rows]

    async def count_documents(self) -> int:
        row = await self.db.fetch_one("SELECT COUNT(*) AS n FROM documents")
        return int(row["n"]) if row else 0

    async def count_chunks(self) -> int:
        row = await self.db.fetch_one("SELECT COUNT(*) AS n FROM chunks")
        return int(row["n"]) if row else 0

    async def count_stale_chunks(
        self, provider: str, model: str, dim: int
    ) -> int:
        row = await self.db.fetch_one(
            """
            SELECT COUNT(*) AS n
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE d.embedding_provider != ?
               OR d.embedding_model != ?
               OR d.embedding_dim != ?
            """,
            (provider, model, dim),
        )
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    # Sessions & messages
    # ------------------------------------------------------------------
    async def create_session(
        self, session_id: str, title: str, customer_id: str
    ) -> dict[str, Any]:
        now = utc_now()
        await self.db.execute(
            """
            INSERT INTO sessions (id, title, customer_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, title, customer_id, now, now),
        )
        return {
            "id": session_id,
            "title": title,
            "customer_id": customer_id,
            "created_at": now,
            "updated_at": now,
        }

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        return dict(row) if row else None

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """
            SELECT s.id, s.title, s.customer_id, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id)
                       AS message_count
            FROM sessions s
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in rows]

    async def touch_session(self, session_id: str) -> None:
        await self.db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (utc_now(), session_id)
        )

    async def delete_session(self, session_id: str) -> bool:
        changed = await self.db.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
        return changed > 0

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        message_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        msg_id = message_id or _new_id()
        now = utc_now()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        async with self.db.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (msg_id, session_id, role, content, meta_json, now),
            )
            await conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )
        return msg_id

    async def get_message(self, message_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """
            SELECT m.*, f.rating AS feedback_rating, f.comment AS feedback_comment
            FROM messages m
            LEFT JOIN feedback f ON f.message_id = m.id
            WHERE m.id = ?
            """,
            (message_id,),
        )
        return self._message_row(row) if row else None

    async def list_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT m.*, f.rating AS feedback_rating, f.comment AS feedback_comment
            FROM messages m
            LEFT JOIN feedback f ON f.message_id = m.id
            WHERE m.session_id = ?
            ORDER BY m.created_at ASC, m.rowid ASC
        """
        params: list[Any] = [session_id]
        if limit is not None:
            # Fetch the newest N, then return them in chronological order.
            sql = """
                SELECT * FROM (
                    SELECT m.rowid AS rowid, m.*,
                           f.rating AS feedback_rating,
                           f.comment AS feedback_comment
                    FROM messages m
                    LEFT JOIN feedback f ON f.message_id = m.id
                    WHERE m.session_id = ?
                    ORDER BY m.created_at DESC, m.rowid DESC
                    LIMIT ?
                ) sub ORDER BY created_at ASC, rowid ASC
            """
            params.append(limit)
        rows = await self.db.fetch_all(sql, params)
        return [self._message_row(r) for r in rows]

    async def update_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[Any] = []
        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if meta is not None:
            sets.append("meta_json = ?")
            params.append(json.dumps(meta, ensure_ascii=False))
        if not sets:
            return
        params.append(message_id)
        await self.db.execute(
            f"UPDATE messages SET {', '.join(sets)} WHERE id = ?", params
        )

    @staticmethod
    def _message_row(row: Any) -> dict[str, Any]:
        item = dict(row)
        try:
            item["meta"] = json.loads(item.get("meta_json") or "{}")
        except json.JSONDecodeError:
            item["meta"] = {}
        rating = item.get("feedback_rating")
        item["feedback"] = (
            {"rating": rating, "comment": item.get("feedback_comment") or ""}
            if rating is not None
            else None
        )
        return item

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    async def upsert_feedback(
        self, message_id: str, rating: int, comment: str = ""
    ) -> dict[str, Any]:
        now = utc_now()
        feedback_id = _new_id()
        await self.db.execute(
            """
            INSERT INTO feedback (id, message_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                rating = excluded.rating,
                comment = excluded.comment,
                created_at = excluded.created_at
            """,
            (feedback_id, message_id, rating, comment, now),
        )
        row = await self.db.fetch_one(
            "SELECT * FROM feedback WHERE message_id = ?", (message_id,)
        )
        return dict(row) if row else {}

    async def get_feedback(self, message_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM feedback WHERE message_id = ?", (message_id,)
        )
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Orders, session facts & tickets
    # ------------------------------------------------------------------
    async def get_order(self, order_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        )
        return dict(row) if row else None

    async def upsert_orders(self, orders: list[dict[str, Any]]) -> int:
        if not orders:
            return 0
        async with self.db.transaction() as conn:
            await conn.executemany(
                """
                INSERT INTO orders
                    (order_id, customer_id, customer_name, status, item_name,
                     amount_cents, ordered_at, tracking_no, logistics_status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    customer_name = excluded.customer_name,
                    status = excluded.status,
                    item_name = excluded.item_name,
                    amount_cents = excluded.amount_cents,
                    ordered_at = excluded.ordered_at,
                    tracking_no = excluded.tracking_no,
                    logistics_status = excluded.logistics_status,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        o["order_id"],
                        o["customer_id"],
                        o["customer_name"],
                        o["status"],
                        o["item_name"],
                        o["amount_cents"],
                        o["ordered_at"],
                        o.get("tracking_no", ""),
                        o.get("logistics_status", ""),
                        o.get("updated_at", utc_now()),
                    )
                    for o in orders
                ],
            )
        return len(orders)

    async def count_orders(self) -> int:
        row = await self.db.fetch_one("SELECT COUNT(*) AS n FROM orders")
        return int(row["n"]) if row else 0

    async def record_session_order(self, session_id: str, order_id: str) -> None:
        await self.db.execute(
            """
            INSERT INTO session_orders (session_id, order_id, checked_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id, order_id) DO UPDATE SET checked_at = excluded.checked_at
            """,
            (session_id, order_id, utc_now()),
        )

    async def has_session_order(self, session_id: str, order_id: str) -> bool:
        row = await self.db.fetch_one(
            "SELECT 1 AS ok FROM session_orders WHERE session_id = ? AND order_id = ?",
            (session_id, order_id),
        )
        return row is not None

    async def count_session_tickets(self, session_id: str) -> int:
        row = await self.db.fetch_one(
            "SELECT COUNT(*) AS n FROM tickets WHERE session_id = ?", (session_id,)
        )
        return int(row["n"]) if row else 0

    async def get_ticket_by_idempotency(self, key: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM tickets WHERE idempotency_key = ?", (key,)
        )
        return dict(row) if row else None

    async def insert_ticket(
        self,
        *,
        ticket_id: str,
        session_id: str,
        order_id: str,
        summary: str,
        priority: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        now = utc_now()
        await self.db.execute(
            """
            INSERT INTO tickets
                (ticket_id, session_id, order_id, summary, priority, status,
                 idempotency_key, created_at)
            VALUES (?, ?, ?, ?, ?, 'open', ?, ?)
            """,
            (ticket_id, session_id, order_id, summary, priority, idempotency_key, now),
        )
        return {
            "ticket_id": ticket_id,
            "session_id": session_id,
            "order_id": order_id,
            "summary": summary,
            "priority": priority,
            "status": "open",
            "idempotency_key": idempotency_key,
            "created_at": now,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    async def stats(
        self, embedding_provider: str, embedding_model: str, embedding_dim: int
    ) -> dict[str, Any]:
        async with self.db.connection() as db:
            async def scalar(sql: str, params: tuple[Any, ...] = ()) -> int:
                async with db.execute(sql, params) as cur:
                    row = await cur.fetchone()
                    return int(row[0]) if row else 0

            documents = await scalar("SELECT COUNT(*) FROM documents")
            chunks = await scalar("SELECT COUNT(*) FROM chunks")
            sessions = await scalar("SELECT COUNT(*) FROM sessions")
            messages = await scalar("SELECT COUNT(*) FROM messages")
            positive = await scalar("SELECT COUNT(*) FROM feedback WHERE rating = 1")
            negative = await scalar("SELECT COUNT(*) FROM feedback WHERE rating = -1")
            tickets = await scalar("SELECT COUNT(*) FROM tickets")
            if embedding_dim > 0:
                stale = await scalar(
                    """
                    SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.doc_id
                    WHERE d.embedding_provider != ? OR d.embedding_model != ?
                       OR d.embedding_dim != ?
                    """,
                    (embedding_provider, embedding_model, embedding_dim),
                )
            else:
                # Remote embedding dimension is discovered on first use; only
                # compare provider/model until then.
                stale = await scalar(
                    """
                    SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id = c.doc_id
                    WHERE d.embedding_provider != ? OR d.embedding_model != ?
                    """,
                    (embedding_provider, embedding_model),
                )
            async with db.execute(
                "SELECT meta_json FROM messages WHERE role = 'assistant'"
            ) as cur:
                meta_rows = await cur.fetchall()

        latencies: list[int] = []
        for row in meta_rows:
            try:
                meta = json.loads(row["meta_json"] or "{}")
            except json.JSONDecodeError:
                continue
            value = meta.get("latency_ms")
            if isinstance(value, (int, float)):
                latencies.append(int(value))
        latencies.sort()

        def percentile(values: list[int], pct: float) -> int | None:
            if not values:
                return None
            index = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
            return values[index]

        return {
            "documents": documents,
            "chunks": chunks,
            "stale_chunks": stale,
            "sessions": sessions,
            "messages": messages,
            "tickets": tickets,
            "feedback_positive": positive,
            "feedback_negative": negative,
            "feedback_total": positive + negative,
            "feedback_positive_rate": (
                round(positive / (positive + negative), 3)
                if positive + negative
                else None
            ),
            "latency_p50_ms": percentile(latencies, 0.5),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_samples": len(latencies),
        }


__all__ = ["Repository", "utc_now"]
