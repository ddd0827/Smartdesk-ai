from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.deps import build_container  # noqa: E402


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Avoid the app's automatic seed pass so the command reports one clear
    # import result (8 documents) instead of auto-seed + force-seed.
    settings = Settings(auto_seed=False)
    container = await build_container(settings)
    try:
        result = await container.documents.seed(force=True)
        print(f"种子数据完成：{result.summary}")
        documents = await container.repo.list_documents()
        print(f"知识库：{len(documents)} 篇文档")
        for doc in documents:
            print(
                f"  - {doc['title']}（{doc['chunk_count']} 个片段，"
                f"{doc['embedding_provider']}:{doc['embedding_model']}）"
            )
        print(f"订单：{await container.repo.count_orders()} 条")
        async with container.db.connection() as db, db.execute(
            "SELECT ticket_id, order_id, priority, status FROM tickets "
            "ORDER BY created_at DESC LIMIT 5"
        ) as cursor:
            tickets = await cursor.fetchall()
        if tickets:
            print(f"最近工单：{len(tickets)} 条")
            for ticket in tickets:
                print(
                    f"  - {ticket['ticket_id']} ({ticket['priority']}, "
                    f"{ticket['status']})"
                )
        return 0
    finally:
        await container.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
