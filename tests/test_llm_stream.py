from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.llm.base import (
    EndEvent,
    LLMError,
    TokenEvent,
    ToolCallsEvent,
    UsageEvent,
)
from app.llm.deepseek import DeepSeekClient, StreamAccumulator


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        auto_seed=False,
        llm_provider="deepseek",
        llm_base_url="https://example.test",
        llm_api_key=SecretStr("test-key"),
        llm_model="deepseek-v4-flash",
        llm_thinking="disabled",
        llm_max_retries=1,
        embedding_provider="hash",
    )
    values.update(overrides)
    return Settings(**values)


def sse_body(payloads: list[dict]) -> bytes:
    body = "".join(
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        for payload in payloads
    )
    return (body + "data: [DONE]\n\n").encode()


def test_stream_accumulator_assembles_split_tool_calls() -> None:
    accumulator = StreamAccumulator()
    accumulator.add_tool_call_deltas(
        [
            {
                "index": 0,
                "id": "call_1",
                "function": {"name": "lookup_order", "arguments": '{"order_'},
            }
        ]
    )
    accumulator.add_tool_call_deltas(
        [{"index": 0, "function": {"arguments": 'id": "A2026001"}'}}]
    )
    accumulator.add_tool_call_deltas(
        [
            {
                "index": 1,
                "id": "call_2",
                "function": {"name": "search_knowledge_base", "arguments": "{}"},
            }
        ]
    )
    calls = accumulator.tool_calls()
    assert [call.name for call in calls] == [
        "lookup_order",
        "search_knowledge_base",
    ]
    assert calls[0].parsed_arguments() == {"order_id": "A2026001"}
    assert calls[1].id == "call_2"


async def test_client_streams_text_and_usage(tmp_path: Path) -> None:
    payloads = [
        {"choices": [{"delta": {"content": "你"}}]},
        {"choices": [{"delta": {"content": "好"}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(payloads),
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekClient(
        make_settings(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    events = [
        event
        async for event in client.stream_chat(
            [{"role": "user", "content": "hi"}]
        )
    ]
    await client.close()
    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "你好"
    assert any(isinstance(e, UsageEvent) for e in events)
    assert isinstance(events[-1], EndEvent)
    assert events[-1].finish_reason == "stop"


async def test_client_assembles_tool_calls_from_stream(tmp_path: Path) -> None:
    payloads = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "lookup_order",
                                    "arguments": '{"order_',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": 'id": "A2026001"}'
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(payloads),
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekClient(
        make_settings(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    events = [
        event
        async for event in client.stream_chat(
            [{"role": "user", "content": "查订单"}],
            tools=[{"type": "function", "function": {"name": "lookup_order"}}],
        )
    ]
    await client.close()
    tool_event = next(e for e in events if isinstance(e, ToolCallsEvent))
    assert tool_event.calls[0].parsed_arguments() == {"order_id": "A2026001"}
    assert isinstance(events[-1], EndEvent)
    assert events[-1].finish_reason == "tool_calls"


async def test_client_maps_auth_errors_without_leaking_key(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid key"}})

    client = DeepSeekClient(
        make_settings(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMError) as exc_info:
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    await client.close()
    assert "API Key" in str(exc_info.value)
    assert "test-key" not in str(exc_info.value)


async def test_client_retries_5xx_then_succeeds(tmp_path: Path) -> None:
    calls = {"count": 0}
    payloads = [
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(
            200,
            content=sse_body(payloads),
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekClient(
        make_settings(tmp_path, llm_max_retries=1),
        transport=httpx.MockTransport(handler),
    )
    text = ""
    async for event in client.stream_chat([{"role": "user", "content": "hi"}]):
        if isinstance(event, TokenEvent):
            text += event.text
    await client.close()
    assert text == "ok"
    assert calls["count"] == 2


async def test_client_retries_without_thinking_when_rejected(tmp_path: Path) -> None:
    calls = {"count": 0}
    payloads = [
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        body = json.loads(request.content.decode("utf-8"))
        if calls["count"] == 1:
            assert body["thinking"] == {"type": "disabled"}
            return httpx.Response(
                400,
                json={"error": {"message": "thinking is not supported"}},
            )
        assert "thinking" not in body
        return httpx.Response(
            200,
            content=sse_body(payloads),
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekClient(
        make_settings(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    text = ""
    async for event in client.stream_chat([{"role": "user", "content": "hi"}]):
        if isinstance(event, TokenEvent):
            text += event.text
    await client.close()
    assert text == "ok"
    assert calls["count"] == 2


async def test_client_raises_on_resource_exhaustion(tmp_path: Path) -> None:
    payloads = [
        {
            "choices": [
                {"delta": {}, "finish_reason": "insufficient_system_resource"}
            ]
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse_body(payloads),
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekClient(
        make_settings(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMError, match="资源不足"):
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    await client.close()


async def test_client_raises_on_empty_response(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client = DeepSeekClient(
        make_settings(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMError, match="没有返回任何内容"):
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    await client.close()
