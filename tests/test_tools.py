from __future__ import annotations

import asyncio
import json

import httpx
from pydantic import BaseModel

from app.rag.index import RetrievedChunk
from app.tools.base import BaseTool, SourceRegistry, ToolContext, ToolResult
from app.tools.kb import SearchKnowledgeBaseArgs, SearchKnowledgeBaseTool
from app.tools.orders import LookupOrderArgs, LookupOrderTool
from app.tools.registry import ToolRegistry
from app.tools.tickets import CreateTicketArgs, CreateTicketTool


def make_chunk(chunk_id: str = "c1", title: str = "退换货政策") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        seq=0,
        title=title,
        filename="01.md",
        section="政策",
        content="七天无理由退货",
        score=0.03,
    )


def test_source_registry_deduplicates_and_assigns_ids() -> None:
    registry = SourceRegistry()
    assert registry.add(make_chunk()) == "S1"
    assert registry.add(make_chunk()) == "S1"
    assert registry.add(make_chunk("c2", "物流")) == "S2"
    snapshot = registry.snapshot()
    assert [item["id"] for item in snapshot] == ["S1", "S2"]
    assert snapshot[0]["title"] == "退换货政策"


async def test_kb_tool_returns_sources(seeded) -> None:
    registry = SourceRegistry()
    context = ToolContext("s1", "C001", registry)
    tool = SearchKnowledgeBaseTool(seeded.index)
    result = await tool.run(
        SearchKnowledgeBaseArgs(query="七天无理由退货", top_k=3), context
    )
    payload = json.loads(result.content)
    assert result.ok is True
    assert payload["results"]
    assert payload["results"][0]["source"] == "S1"
    assert result.meta["source_ids"] == [item["source"] for item in payload["results"]]
    assert len(registry.snapshot()) == len(payload["results"])


async def test_order_tool_enforces_customer_scope(seeded) -> None:
    repo = seeded.repo
    await repo.create_session("s1", "测试", "C001")
    context = ToolContext("s1", "C001", SourceRegistry())
    tool = LookupOrderTool(repo)

    result = await tool.run(LookupOrderArgs(order_id="A2026001"), context)
    payload = json.loads(result.content)
    assert result.ok is True
    assert payload["order"]["status"] == "已发货"
    assert await repo.has_session_order("s1", "A2026001") is True

    forbidden = await tool.run(LookupOrderArgs(order_id="A2026011"), context)
    assert forbidden.ok is False
    assert "无权" in forbidden.content or "未找到" in forbidden.content
    assert await repo.has_session_order("s1", "A2026011") is False


async def test_ticket_tool_requires_prior_order_lookup(seeded) -> None:
    repo = seeded.repo
    await repo.create_session("s1", "测试", "C001")
    tool = CreateTicketTool(
        repo,
        base_url="http://testserver",
        api_key="demo-itsm-key",
    )
    context = ToolContext("s1", "C001", SourceRegistry())
    result = await tool.run(
        CreateTicketArgs(
            order_id="A2026001", summary="物流三天没有更新，需要人工跟进"
        ),
        context,
    )
    await tool.close()
    assert result.ok is False
    assert "先查询" in result.content


async def test_ticket_tool_posts_with_auth_and_idempotency(seeded) -> None:
    repo = seeded.repo
    await repo.create_session("s-ticket", "测试", "C001")
    context = ToolContext("s-ticket", "C001", SourceRegistry())
    order_tool = LookupOrderTool(repo)
    await order_tool.run(LookupOrderArgs(order_id="A2026001"), context)

    captured: dict[str, str] = {}
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        captured["api_key"] = request.headers["X-API-Key"]
        captured["idempotency_key"] = request.headers["Idempotency-Key"]
        return httpx.Response(
            201,
            json={
                "ticket_id": "TMOCK001",
                "status": "open",
                "priority": "normal",
                "duplicate": False,
            },
        )

    tool = CreateTicketTool(
        repo,
        base_url="http://testserver",
        api_key="demo-itsm-key",
        transport=httpx.MockTransport(handler),
    )
    args = CreateTicketArgs(
        order_id="A2026001", summary="物流三天没有更新，需要人工跟进"
    )
    first = await tool.run(args, context)
    assert first.ok is True
    assert json.loads(first.content)["ticket"]["ticket_id"] == "TMOCK001"
    assert captured["api_key"] == "demo-itsm-key"
    assert captured["idempotency_key"]

    # The real mock-ITSM endpoint persists the ticket; simulate that here so
    # the second call can be served from the local idempotency record.
    await repo.insert_ticket(
        ticket_id="TMOCK001",
        session_id="s-ticket",
        order_id="A2026001",
        summary=args.summary,
        priority=args.priority,
        idempotency_key=captured["idempotency_key"],
    )
    second = await tool.run(args, context)
    assert second.meta["duplicate"] is True
    assert calls["count"] == 1
    await tool.close()


async def test_ticket_tool_auth_failure_is_not_retried(seeded) -> None:
    repo = seeded.repo
    await repo.create_session("s-auth", "测试", "C001")
    context = ToolContext("s-auth", "C001", SourceRegistry())
    await LookupOrderTool(repo).run(
        LookupOrderArgs(order_id="A2026001"), context
    )
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, json={"detail": "bad key"})

    tool = CreateTicketTool(
        repo,
        base_url="http://testserver",
        api_key="wrong",
        transport=httpx.MockTransport(handler),
    )
    result = await tool.run(
        CreateTicketArgs(order_id="A2026001", summary="需要人工处理物流问题"),
        context,
    )
    await tool.close()
    assert result.ok is False
    assert "鉴权失败" in result.content
    assert calls["count"] == 1


async def test_ticket_tool_retries_5xx_then_reports_unavailable(
    seeded, monkeypatch
) -> None:
    repo = seeded.repo
    await repo.create_session("s-retry", "测试", "C001")
    context = ToolContext("s-retry", "C001", SourceRegistry())
    await LookupOrderTool(repo).run(
        LookupOrderArgs(order_id="A2026001"), context
    )
    calls = {"count": 0}

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.tools.tickets.asyncio.sleep", no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503, json={"detail": "down"})

    tool = CreateTicketTool(
        repo,
        base_url="http://testserver",
        api_key="demo-itsm-key",
        transport=httpx.MockTransport(handler),
    )
    result = await tool.run(
        CreateTicketArgs(order_id="A2026001", summary="物流异常需要人工处理"),
        context,
    )
    await tool.close()
    assert result.ok is False
    assert "暂时不可用" in result.content
    assert calls["count"] == 3


async def test_ticket_tool_honors_session_limit(seeded) -> None:
    repo = seeded.repo
    await repo.create_session("s-limit", "测试", "C001")
    await repo.record_session_order("s-limit", "A2026001")
    for index in range(3):
        await repo.insert_ticket(
            ticket_id=f"T{index}",
            session_id="s-limit",
            order_id="A2026001",
            summary="已有工单",
            priority="normal",
            idempotency_key=f"key-{index}",
        )
    tool = CreateTicketTool(
        repo,
        base_url="http://testserver",
        api_key="demo-itsm-key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(201, json={"ticket_id": "X"})
        ),
    )
    result = await tool.run(
        CreateTicketArgs(order_id="A2026001", summary="再建一张工单"),
        ToolContext("s-limit", "C001", SourceRegistry()),
    )
    await tool.close()
    assert result.ok is False
    assert "达到上限" in result.content


class EmptyArgs(BaseModel):
    pass


class SlowTool(BaseTool):
    name = "slow"
    description = "slow"
    args_model = EmptyArgs

    async def run(self, args, ctx) -> ToolResult:
        await asyncio.sleep(1)
        return ToolResult(True, "done", "done")


class LongTool(BaseTool):
    name = "long"
    description = "long"
    args_model = EmptyArgs

    async def run(self, args, ctx) -> ToolResult:
        return ToolResult(True, "x" * 200, "long result")


async def test_registry_validates_and_truncates(seeded) -> None:
    registry = ToolRegistry(
        [LookupOrderTool(seeded.repo), SlowTool(), LongTool()],
        max_result_chars=20,
        timeout_seconds=0.05,
    )
    context = ToolContext("s1", "C001", SourceRegistry())
    unknown, _ = await registry.execute("nope", "{}", context)
    assert unknown.ok is False
    bad_json, _ = await registry.execute("slow", "{bad", context)
    assert bad_json.ok is False
    invalid, _ = await registry.execute(
        "lookup_order", json.dumps({"order_id": "!!"}), context
    )
    assert invalid.ok is False
    assert invalid.meta["reason"] == "validation"
    timeout, _ = await registry.execute("slow", "{}", context)
    assert timeout.ok is False
    assert timeout.meta["reason"] == "timeout"
    long_result, _ = await registry.execute("long", "{}", context)
    assert len(long_result.content) < 60
    assert "截断" in long_result.content
    assert {schema["function"]["name"] for schema in registry.schemas()} == {
        "lookup_order",
        "slow",
        "long",
    }
