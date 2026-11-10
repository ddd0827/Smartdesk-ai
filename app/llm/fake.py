from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from app.llm.base import (
    EndEvent,
    LLMError,
    LLMEvent,
    TokenEvent,
    ToolCall,
    ToolCallsEvent,
    UsageEvent,
)

_ORDER_ID_RE = re.compile(r"\b(A\d{6,})\b", re.IGNORECASE)
_POLICY_KEYWORDS = (
    "退",
    "换",
    "保修",
    "维修",
    "物流",
    "快递",
    "丢件",
    "破损",
    "发票",
    "退款",
    "优惠券",
    "积分",
    "会员",
    "投诉",
    "安装",
    "故障",
    "企业",
    "采购",
    "折扣",
    "政策",
    "多久",
    "怎么",
    "能否",
    "可以吗",
)
_TICKET_KEYWORDS = ("工单", "人工", "投诉", "升级", "主管")


def _json_content(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_tool_content(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


class FakeLLM:
    """Deterministic mock model for tests and the offline UI demo.

    Two modes:
    - scripted: consumes a queue of event lists (unit/integration tests)
    - auto: inspects the conversation and emits plausible tool calls/answers
      so the whole UI can run without an API key.
    """

    model = "mock"

    def __init__(
        self,
        scripts: list[list[LLMEvent]] | None = None,
        *,
        delay_seconds: float = 0.0,
    ) -> None:
        self._scripts = list(scripts) if scripts is not None else None
        self.delay_seconds = delay_seconds
        self.calls = 0
        self._tool_counter = 0

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        if self._scripts is not None:
            if not self._scripts:
                raise LLMError("FakeLLM 脚本已用完")
            script = list(self._scripts.pop(0))
            if not script or not isinstance(script[-1], EndEvent):
                script.append(EndEvent(finish_reason="stop"))
            for event in script:
                if self.delay_seconds:
                    await asyncio.sleep(self.delay_seconds)
                yield event
            return

        for event in self._auto_events(messages):
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            yield event

    # ------------------------------------------------------------------
    # Auto / demo mode
    # ------------------------------------------------------------------
    def _next_call_id(self) -> str:
        self._tool_counter += 1
        return f"call_mock_{self._tool_counter}"

    def _tool_call(self, name: str, arguments: dict[str, Any]) -> ToolCallsEvent:
        return ToolCallsEvent(
            calls=[
                ToolCall(
                    id=self._next_call_id(),
                    name=name,
                    arguments=_json_content(arguments),
                )
            ]
        )

    def _current_turn(self, messages: list[dict[str, Any]]) -> tuple[str, list[str], list[dict]]:
        last_user_index = 0
        for index, message in enumerate(messages):
            if message.get("role") == "user":
                last_user_index = index
        user_text = str(messages[last_user_index].get("content") or "")
        executed: list[str] = []
        tool_messages: list[dict[str, Any]] = []
        for message in messages[last_user_index + 1 :]:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    name = (call.get("function") or {}).get("name")
                    if name:
                        executed.append(name)
            elif message.get("role") == "tool":
                tool_messages.append(message)
        return user_text, executed, tool_messages

    def _auto_events(self, messages: list[dict[str, Any]]) -> list[LLMEvent]:
        user_text, executed, tool_messages = self._current_turn(messages)
        order_match = _ORDER_ID_RE.search(user_text)
        order_id = order_match.group(1).upper() if order_match else None
        wants_policy = any(keyword in user_text for keyword in _POLICY_KEYWORDS)
        wants_ticket = any(keyword in user_text for keyword in _TICKET_KEYWORDS)

        lookup_ok = False
        for message in tool_messages:
            payload = _parse_tool_content(str(message.get("content") or ""))
            if payload.get("ok") and payload.get("order"):
                lookup_ok = True

        if order_id and "lookup_order" not in executed:
            return [
                self._tool_call("lookup_order", {"order_id": order_id}),
                EndEvent(finish_reason="tool_calls"),
            ]

        if wants_policy and "search_knowledge_base" not in executed:
            return [
                self._tool_call(
                    "search_knowledge_base",
                    {"query": user_text[:160], "top_k": 4},
                ),
                EndEvent(finish_reason="tool_calls"),
            ]

        if (
            wants_ticket
            and order_id
            and lookup_ok
            and "create_ticket" not in executed
        ):
            priority = "high" if ("投诉" in user_text or "升级" in user_text) else "normal"
            return [
                self._tool_call(
                    "create_ticket",
                    {
                        "order_id": order_id,
                        "summary": user_text[:300],
                        "priority": priority,
                    },
                ),
                EndEvent(finish_reason="tool_calls"),
            ]

        answer = self._compose_answer(user_text, tool_messages)
        return [
            *[TokenEvent(answer[index : index + 14]) for index in range(0, len(answer), 14)],
            UsageEvent(
                prompt_tokens=sum(len(str(m.get("content") or "")) for m in messages) // 2,
                completion_tokens=len(answer) // 2,
                total_tokens=(
                    sum(len(str(m.get("content") or "")) for m in messages) // 2
                    + len(answer) // 2
                ),
            ),
            EndEvent(finish_reason="stop"),
        ]

    def _compose_answer(
        self, user_text: str, tool_messages: list[dict[str, Any]]
    ) -> str:
        if not tool_messages:
            return (
                "（离线演示模式）你好，我是智服，悦家电器售后客服。"
                "你可以问我退换货、物流、保修、发票等问题，"
                "也可以提供订单号查询进度。"
            )
        lines: list[str] = ["（离线演示模式）我根据知识库和订单数据为你整理如下："]
        sources: list[dict[str, Any]] = []
        order: dict[str, Any] | None = None
        ticket: dict[str, Any] | None = None
        error: str | None = None

        for message in tool_messages:
            payload = _parse_tool_content(str(message.get("content") or ""))
            if payload.get("order"):
                order = payload["order"]
            if payload.get("results"):
                sources = payload["results"]
            if payload.get("ticket"):
                ticket = payload["ticket"]
            if payload.get("ok") is False and payload.get("error"):
                error = str(payload["error"])

        if order:
            lines.append(
                f"\n订单 {order.get('order_id')}：状态为「{order.get('status')}」，"
                f"商品是 {order.get('item_name')}，"
                f"物流信息：{order.get('logistics_status') or '暂无'}。"
            )
        if sources:
            for index, source in enumerate(sources[:2], start=1):
                content = " ".join(str(source.get("content") or "").split())
                if len(content) > 130:
                    content = f"{content[:130]}…"
                lines.append(
                    f"\n{index}. {content}"
                    f"（{source.get('title')}，来源 "
                    f"[{source.get('source', f'S{index}')}]）"
                )
            lines.append("\n以上政策信息来自知识库，具体处理结果以客服审核为准。")
        if ticket:
            lines.append(
                f"\n已为你创建工单 {ticket.get('ticket_id')}，优先级为 {ticket.get('priority')}。"
                "客服会在 24 小时内联系你，请保持电话畅通。"
            )
        if error and not order:
            lines.append(f"\n很抱歉，{error}。")
        if not order and not sources and not ticket and not error:
            lines.append(
                "\n我暂时没有检索到相关信息。你可以补充订单号，或描述具体问题，我再帮你查询。"
            )
        lines.append("\n如果你需要人工跟进，可以直接回复「帮我建工单」。")
        return "".join(lines)

    async def close(self) -> None:
        return None


__all__ = ["FakeLLM"]
