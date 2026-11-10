from __future__ import annotations

import numpy as np

from app.store.db import Database
from app.store.repo import Repository


async def test_session_messages_and_feedback(repo: Repository) -> None:
    session_id = "session-1"
    await repo.create_session(session_id, "测试会话", "C001")
    user_id = await repo.add_message(session_id, "user", "你好")
    assistant_id = await repo.add_message(
        session_id,
        "assistant",
        "您好，有什么可以帮您？",
        meta={"latency_ms": 120, "trace": [{"tool": "lookup_order"}]},
    )

    sessions = await repo.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["message_count"] == 2

    messages = await repo.list_messages(session_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["id"] == assistant_id
    assert messages[1]["meta"]["latency_ms"] == 120

    await repo.upsert_feedback(assistant_id, 1, "有帮助")
    detail = await repo.get_message(assistant_id)
    assert detail is not None
    assert detail["feedback"]["rating"] == 1
    assert detail["feedback"]["comment"] == "有帮助"

    stats = await repo.stats("hash", "hash-ngram-128", 128)
    assert stats["messages"] == 2
    assert stats["feedback_positive"] == 1
    assert stats["feedback_positive_rate"] == 1.0
    assert stats["latency_p50_ms"] == 120

    assert await repo.delete_session(session_id) is True
    assert await repo.list_messages(session_id) == []
    assert user_id


async def test_document_delete_cascades_chunks(repo: Repository) -> None:
    doc_id = "doc-1"
    embedding = np.zeros(4, dtype=np.float32).tobytes()
    doc = {
        "id": doc_id,
        "title": "测试文档",
        "filename": "test.md",
        "content_hash": "hash-1",
        "raw_text": "正文",
        "size_bytes": 6,
        "embedding_provider": "hash",
        "embedding_model": "hash-ngram-128",
        "embedding_dim": 4,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    chunks = [
        {
            "id": f"{doc_id}#0",
            "doc_id": doc_id,
            "seq": 0,
            "section": "A",
            "content": "正文",
            "embedding": embedding,
            "created_at": doc["created_at"],
        }
    ]
    await repo.insert_document(doc, chunks)
    assert await repo.count_chunks() == 1
    assert await repo.find_document_by_hash("hash-1")

    assert await repo.delete_document(doc_id) is True
    assert await repo.count_documents() == 0
    assert await repo.count_chunks() == 0


async def test_orders_session_facts_and_tickets(repo: Repository) -> None:
    await repo.create_session("session-2", "订单会话", "C001")
    await repo.upsert_orders(
        [
            {
                "order_id": "A2026001",
                "customer_id": "C001",
                "customer_name": "林晓",
                "status": "已发货",
                "item_name": "净水器",
                "amount_cents": 189900,
                "ordered_at": "2026-11-01",
                "tracking_no": "SF1",
                "logistics_status": "运输中",
                "updated_at": "2026-11-09T10:00:00+08:00",
            }
        ]
    )
    order = await repo.get_order("A2026001")
    assert order is not None
    assert order["status"] == "已发货"

    assert await repo.has_session_order("session-2", "A2026001") is False
    await repo.record_session_order("session-2", "A2026001")
    assert await repo.has_session_order("session-2", "A2026001") is True

    ticket = await repo.insert_ticket(
        ticket_id="T001",
        session_id="session-2",
        order_id="A2026001",
        summary="物流异常",
        priority="high",
        idempotency_key="key-1",
    )
    assert ticket["ticket_id"] == "T001"
    assert await repo.count_session_tickets("session-2") == 1
    existing = await repo.get_ticket_by_idempotency("key-1")
    assert existing is not None
    assert existing["priority"] == "high"


async def test_transaction_rolls_back_on_error(db: Database) -> None:
    try:
        async with db.transaction() as connection:
            await connection.execute(
                "INSERT INTO orders(order_id, customer_id, customer_name, status, "
                "item_name, amount_cents, ordered_at, tracking_no, logistics_status, "
                "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("X1", "C1", "A", "s", "i", 1, "d", "", "", "u"),
            )
            raise RuntimeError("rollback")
    except RuntimeError:
        pass
    row = await db.fetch_one("SELECT * FROM orders WHERE order_id = ?", ("X1",))
    assert row is None
