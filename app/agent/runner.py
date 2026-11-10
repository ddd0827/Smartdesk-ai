from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from app.agent.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    TokenDelta,
    TraceEvent,
)
from app.agent.prompts import build_system_prompt
from app.config import Settings
from app.llm.base import (
    EndEvent,
    LLMClient,
    LLMError,
    TokenEvent,
    ToolCall,
    ToolCallsEvent,
    UsageEvent,
)
from app.store.repo import Repository
from app.tools.base import SourceRegistry, ToolContext, ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_MAX_HISTORY_CHARS = 6000
_MAX_HISTORY_MESSAGE_CHARS = 1200


class AgentRunner:
    """Tool-calling loop with persistence, citations and cancellation handling."""

    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient,
        tools: ToolRegistry,
        repo: Repository,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.tools = tools
        self.repo = repo

    async def run(
        self,
        *,
        session_id: str,
        customer_id: str,
        message_id: str,
        user_text: str,
    ) -> AsyncIterator[AgentEvent]:
        started = time.perf_counter()
        if await self.repo.get_message(message_id) is None:
            await self.repo.add_message(
                session_id,
                "assistant",
                "",
                message_id=message_id,
                meta={"status": "streaming"},
            )

        rows = await self.repo.list_messages(
            session_id, limit=self.settings.history_messages
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(customer_id),
            },
            *self._history_messages(rows, exclude_id=message_id),
        ]
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": user_text})

        tool_schemas = self.tools.schemas()
        sources = SourceRegistry()
        ctx = ToolContext(
            session_id=session_id,
            customer_id=customer_id,
            sources=sources,
        )
        trace: list[dict[str, Any]] = []
        usages: list[UsageEvent] = []
        final_text = ""
        finish_reason = "stop"
        completed = False

        try:
            for step in range(1, self.settings.max_agent_steps + 1):
                round_text = ""
                tool_calls: list[ToolCall] = []
                async for event in self.llm.stream_chat(messages, tool_schemas):
                    if isinstance(event, TokenEvent):
                        round_text += event.text
                        final_text += event.text
                        yield TokenDelta(event.text)
                    elif isinstance(event, ToolCallsEvent):
                        tool_calls = event.calls
                    elif isinstance(event, UsageEvent):
                        usages.append(event)
                    elif isinstance(event, EndEvent):
                        finish_reason = event.finish_reason

                if not tool_calls:
                    completed = True
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": round_text,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )

                for index, call in enumerate(tool_calls):
                    args_preview = self._args_preview(call)
                    yield TraceEvent(
                        phase="start",
                        tool=call.name,
                        step=step,
                        args=args_preview,
                        call_id=call.id,
                    )
                    if index >= self.settings.max_tool_calls_per_step:
                        result = ToolResult(
                            ok=False,
                            content=json.dumps(
                                {
                                    "ok": False,
                                    "error": "单轮工具调用数量超过上限，未执行。",
                                },
                                ensure_ascii=False,
                            ),
                            summary="超过单轮工具调用上限",
                            meta={"reason": "too_many_tool_calls"},
                        )
                        duration_ms = 0.0
                    else:
                        result, duration_ms = await self.tools.execute(
                            call.name, call.arguments, ctx
                        )

                    trace_item = {
                        "tool": call.name,
                        "args": args_preview,
                        "ok": result.ok,
                        "duration_ms": round(duration_ms, 1),
                        "summary": result.summary[:300],
                        "step": step,
                        "call_id": call.id,
                    }
                    trace.append(trace_item)
                    yield TraceEvent(
                        phase="end",
                        tool=call.name,
                        step=step,
                        args=args_preview,
                        ok=result.ok,
                        duration_ms=round(duration_ms, 1),
                        summary=result.summary[:300],
                        call_id=call.id,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result.content,
                        }
                    )
                    if result.meta.get("source_ids"):
                        yield SourcesEvent(sources=sources.snapshot())

            if not completed:
                notice = "\n\n（已达到本次处理的最大步骤数，请补充信息或转人工客服。）"
                if final_text:
                    final_text += notice
                    yield TokenDelta(notice)
                else:
                    final_text = "抱歉，这个问题暂时超出我的处理范围，请补充订单号或转人工客服。"
                    yield TokenDelta(final_text)
                finish_reason = "max_steps"

        except asyncio.CancelledError:
            meta = self._meta(
                trace=trace,
                sources=sources.snapshot(),
                usages=usages,
                started=started,
                finish_reason="cancelled",
                invalid_citations=[],
                interrupted=True,
            )
            # The producer task is being cancelled by a disconnected client.
            # Persist in a new task so the write is not cancelled with us.
            asyncio.create_task(
                self._safe_persist(
                    message_id,
                    final_text or "（回答已中断）",
                    meta,
                )
            )
            raise
        except LLMError as exc:
            logger.warning("Agent LLM error: %s", exc)
            if sources.snapshot():
                yield SourcesEvent(sources=sources.snapshot())
            await self._safe_persist(
                message_id,
                final_text or "抱歉，本次回答生成失败，请稍后重试。",
                self._meta(
                    trace=trace,
                    sources=sources.snapshot(),
                    usages=usages,
                    started=started,
                    finish_reason="error",
                    invalid_citations=[],
                    interrupted=False,
                    error=str(exc)[:500],
                ),
            )
            yield ErrorEvent(message=str(exc), code="llm_error")
            return
        except Exception:
            logger.exception("Agent 处理失败")
            if sources.snapshot():
                yield SourcesEvent(sources=sources.snapshot())
            await self._safe_persist(
                message_id,
                final_text or "抱歉，客服系统处理失败，请稍后重试。",
                self._meta(
                    trace=trace,
                    sources=sources.snapshot(),
                    usages=usages,
                    started=started,
                    finish_reason="error",
                    invalid_citations=[],
                    interrupted=False,
                    error="internal_error",
                ),
            )
            yield ErrorEvent(
                message="客服系统处理失败，请稍后重试。", code="internal_error"
            )
            return

        if not final_text.strip():
            final_text = "抱歉，我没有生成有效回答，请补充信息后重试。"
            yield TokenDelta(final_text)

        source_list = sources.snapshot()
        invalid = self._invalid_citations(final_text, source_list)
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = {
            "prompt_tokens": sum(item.prompt_tokens for item in usages),
            "completion_tokens": sum(item.completion_tokens for item in usages),
            "total_tokens": sum(item.total_tokens for item in usages),
        }
        meta = self._meta(
            trace=trace,
            sources=source_list,
            usages=usages,
            started=started,
            finish_reason=finish_reason,
            invalid_citations=invalid,
            interrupted=False,
        )
        await self._safe_persist(message_id, final_text, meta)
        if source_list:
            yield SourcesEvent(sources=source_list)
        yield DoneEvent(
            message_id=message_id,
            latency_ms=latency_ms,
            usage=usage,
            invalid_citations=invalid,
            interrupted=False,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _history_messages(
        self, rows: list[dict[str, Any]], *, exclude_id: str
    ) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        total = 0
        for row in reversed(rows):
            if row["id"] == exclude_id:
                continue
            if row["role"] not in {"user", "assistant"}:
                continue
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if row["role"] == "assistant" and (row.get("meta") or {}).get("error"):
                continue
            # Source IDs belong to the turn that produced them. Remove them
            # from history so the model cannot cite a stale S-number.
            content = re.sub(r"\[S\d+\]", "", content)
            content = content[:_MAX_HISTORY_MESSAGE_CHARS]
            if selected and total + len(content) > _MAX_HISTORY_CHARS:
                break
            selected.append({"role": row["role"], "content": content})
            total += len(content)
        selected.reverse()
        return selected

    @staticmethod
    def _args_preview(call: ToolCall) -> dict[str, Any]:
        try:
            parsed = json.loads(call.arguments or "{}")
            if isinstance(parsed, dict):
                return {
                    str(key): (str(value)[:300] if not isinstance(value, (int, float)) else value)
                    for key, value in list(parsed.items())[:8]
                }
        except (json.JSONDecodeError, TypeError):
            pass
        return {"_raw": call.arguments[:300]}

    @staticmethod
    def _invalid_citations(
        text: str, sources: list[dict[str, Any]]
    ) -> list[str]:
        valid = {source["id"] for source in sources}
        used = {f"S{number}" for number in _CITATION_RE.findall(text)}
        return sorted(used - valid)

    def _meta(
        self,
        *,
        trace: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        usages: list[UsageEvent],
        started: float,
        finish_reason: str,
        invalid_citations: list[str],
        interrupted: bool,
        error: str | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "trace": trace,
            "sources": sources,
            "usage": {
                "prompt_tokens": sum(item.prompt_tokens for item in usages),
                "completion_tokens": sum(item.completion_tokens for item in usages),
                "total_tokens": sum(item.total_tokens for item in usages),
            },
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "finish_reason": finish_reason,
            "invalid_citations": invalid_citations,
            "interrupted": interrupted,
            "model": self.llm.model,
        }
        if error:
            meta["error"] = error
        return meta

    async def _safe_persist(
        self, message_id: str, content: str, meta: dict[str, Any]
    ) -> None:
        try:
            await self.repo.update_message(message_id, content=content, meta=meta)
        except Exception:
            logger.exception("保存助手消息失败 message_id=%s", message_id)


__all__ = ["AgentRunner"]
