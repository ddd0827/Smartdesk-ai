from __future__ import annotations

import asyncio

import pytest

from app.agent.events import (
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    TokenDelta,
    TraceEvent,
)
from app.config import Settings
from app.deps import build_container
from app.llm.base import (
    EndEvent,
    TokenEvent,
    ToolCall,
    ToolCallsEvent,
    UsageEvent,
)
from app.llm.fake import FakeLLM


async def prepare_session(container, user_text: str) -> tuple[str, str]:
    session_id = "s-agent"
    await container.repo.create_session(session_id, "Agent 测试", "C001")
    await container.repo.add_message(session_id, "user", user_text)
    message_id = "m-agent"
    await container.repo.add_message(
        session_id,
        "assistant",
        "",
        message_id=message_id,
        meta={"status": "streaming"},
    )
    return session_id, message_id


def tool_call(call_id: str, name: str, arguments: str) -> list:
    return [
        ToolCallsEvent(
            calls=[ToolCall(id=call_id, name=name, arguments=arguments)]
        ),
        EndEvent(finish_reason="tool_calls"),
    ]


async def test_agent_tool_loop_and_persistence(
    settings: Settings,
) -> None:
    scripts = [
        tool_call("c1", "lookup_order", '{"order_id":"A2026001"}'),
        tool_call(
            "c2",
            "search_knowledge_base",
            '{"query":"物流 退货 政策","top_k":2}',
        ),
        [
            TokenEvent("物流超过 72 小时未更新属于异常 [S1]。"),
            UsageEvent(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            EndEvent(finish_reason="stop"),
        ],
    ]
    live_settings = settings.model_copy(update={"auto_seed": True})
    container = await build_container(live_settings, llm=FakeLLM(scripts))
    try:
        session_id, message_id = await prepare_session(container, "订单 A2026001 怎么办？")
        events = [
            event
            async for event in container.agent.run(
                session_id=session_id,
                customer_id="C001",
                message_id=message_id,
                user_text="订单 A2026001 怎么办？",
            )
        ]
        trace_events = [event for event in events if isinstance(event, TraceEvent)]
        assert len(trace_events) == 4
        assert trace_events[0].phase == "start"
        assert trace_events[0].tool == "lookup_order"
        assert trace_events[1].ok is True
        assert any(isinstance(event, SourcesEvent) for event in events)
        assert any(isinstance(event, TokenDelta) for event in events)
        done = next(event for event in events if isinstance(event, DoneEvent))
        assert done.finish_reason == "stop"
        assert done.invalid_citations == []
        assert done.usage["total_tokens"] == 120

        stored = await container.repo.get_message(message_id)
        assert stored is not None
        assert "72 小时" in stored["content"]
        assert len(stored["meta"]["trace"]) == 2
        assert stored["meta"]["sources"]
        assert stored["meta"]["interrupted"] is False
    finally:
        await container.close()


async def test_agent_reaches_max_steps(settings: Settings) -> None:
    scripts = [
        tool_call("c1", "lookup_order", '{"order_id":"A2026001"}'),
        tool_call("c2", "search_knowledge_base", '{"query":"退货"}'),
    ]
    live_settings = settings.model_copy(
        update={"auto_seed": True, "max_agent_steps": 2}
    )
    container = await build_container(live_settings, llm=FakeLLM(scripts))
    try:
        session_id, message_id = await prepare_session(container, "查询订单")
        events = [
            event
            async for event in container.agent.run(
                session_id=session_id,
                customer_id="C001",
                message_id=message_id,
                user_text="查询订单",
            )
        ]
        done = next(event for event in events if isinstance(event, DoneEvent))
        assert done.finish_reason == "max_steps"
        stored = await container.repo.get_message(message_id)
        assert stored is not None
        assert stored["content"]
        assert stored["meta"]["finish_reason"] == "max_steps"
    finally:
        await container.close()


async def test_agent_handles_llm_error(settings: Settings) -> None:
    scripts = [tool_call("c1", "lookup_order", '{"order_id":"A2026001"}')]
    live_settings = settings.model_copy(update={"auto_seed": True})
    container = await build_container(live_settings, llm=FakeLLM(scripts))
    try:
        session_id, message_id = await prepare_session(container, "查询订单")
        events = [
            event
            async for event in container.agent.run(
                session_id=session_id,
                customer_id="C001",
                message_id=message_id,
                user_text="查询订单",
            )
        ]
        error = next(event for event in events if isinstance(event, ErrorEvent))
        assert error.code == "llm_error"
        stored = await container.repo.get_message(message_id)
        assert stored is not None
        assert stored["meta"]["error"]
        assert stored["content"]
    finally:
        await container.close()


async def test_agent_flags_invalid_citations(settings: Settings) -> None:
    scripts = [
        [
            TokenEvent("根据政策 [S9]，可以退货。"),
            EndEvent(finish_reason="stop"),
        ]
    ]
    live_settings = settings.model_copy(update={"auto_seed": True})
    container = await build_container(live_settings, llm=FakeLLM(scripts))
    try:
        session_id, message_id = await prepare_session(container, "能退货吗？")
        events = [
            event
            async for event in container.agent.run(
                session_id=session_id,
                customer_id="C001",
                message_id=message_id,
                user_text="能退货吗？",
            )
        ]
        done = next(event for event in events if isinstance(event, DoneEvent))
        assert done.invalid_citations == ["S9"]
        stored = await container.repo.get_message(message_id)
        assert stored is not None
        assert stored["meta"]["invalid_citations"] == ["S9"]
    finally:
        await container.close()


class SlowLLM:
    model = "slow"

    async def stream_chat(self, messages, tools=None):
        yield TokenEvent("部分回答")
        await asyncio.sleep(5)
        yield EndEvent(finish_reason="stop")

    async def close(self) -> None:
        return None


async def test_agent_persists_partial_answer_on_cancellation(
    settings: Settings,
) -> None:
    live_settings = settings.model_copy(update={"auto_seed": True})
    container = await build_container(live_settings, llm=SlowLLM())
    try:
        session_id, message_id = await prepare_session(container, "慢速问题")

        async def consume() -> None:
            async for _ in container.agent.run(
                session_id=session_id,
                customer_id="C001",
                message_id=message_id,
                user_text="慢速问题",
            ):
                pass

        task = asyncio.create_task(consume())
        for _ in range(50):
            stored = await container.repo.get_message(message_id)
            if stored and stored["content"]:
                break
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        stored = None
        for _ in range(50):
            stored = await container.repo.get_message(message_id)
            if stored and stored["meta"].get("interrupted"):
                break
            await asyncio.sleep(0.02)
        assert stored is not None
        assert stored["content"] == "部分回答"
        assert stored["meta"]["interrupted"] is True
        assert stored["meta"]["finish_reason"] == "cancelled"
    finally:
        await container.close()
