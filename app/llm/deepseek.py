from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import Settings
from app.llm.base import (
    EndEvent,
    LLMError,
    LLMEvent,
    TokenEvent,
    ToolCall,
    ToolCallsEvent,
    UsageEvent,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_TRANSPORT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)


@dataclass
class _ToolCallAccumulator:
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamAccumulator:
    """Assemble OpenAI-style streaming deltas.

    Tool calls are keyed by ``index``. ``id`` and ``function.name`` usually
    appear only in the first delta; ``function.arguments`` may be split across
    arbitrary chunk boundaries, so it is concatenated and parsed only after the
    stream finishes.
    """

    text: str = ""
    reasoning_content: str = ""
    finish_reason: str | None = None
    usage: UsageEvent | None = None
    _tool_calls: dict[int, _ToolCallAccumulator] = field(default_factory=dict)
    _bad_tool_arguments: list[int] = field(default_factory=list)

    def add_tool_call_deltas(self, deltas: list[dict[str, Any]]) -> None:
        for delta in deltas:
            index = int(delta.get("index", 0))
            item = self._tool_calls.setdefault(index, _ToolCallAccumulator(index))
            call_id = delta.get("id") or ""
            if call_id and not item.id:
                item.id = call_id
            function = delta.get("function") or {}
            name = function.get("name") or ""
            if name:
                if not item.name:
                    item.name = name
                elif name not in item.name:
                    item.name += name
            arguments = function.get("arguments")
            if arguments:
                item.arguments += arguments

    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(
                id=item.id or f"call_{item.index}",
                name=item.name,
                arguments=item.arguments or "{}",
            )
            for item in sorted(self._tool_calls.values(), key=lambda tc: tc.index)
        ]

    def set_usage(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        self.usage = UsageEvent(
            prompt_tokens=int(payload.get("prompt_tokens") or 0),
            completion_tokens=int(payload.get("completion_tokens") or 0),
            total_tokens=int(payload.get("total_tokens") or 0),
        )


class DeepSeekClient:
    """Minimal OpenAI-compatible streaming client built on httpx.

    We intentionally keep messages as plain dictionaries instead of typed SDK
    objects so non-standard fields (thinking/reasoning_content) are not dropped
    and can be handled explicitly.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=15.0,
                    read=self.settings.llm_timeout_seconds,
                    write=30.0,
                    pool=15.0,
                ),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
                follow_redirects=False,
                transport=self._transport,
            )
        return self._client

    @property
    def endpoint(self) -> str:
        return f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key_value}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools
        thinking = self.settings.llm_thinking
        if tools and thinking == "enabled":
            # The agent loop does not replay reasoning_content on tool-call
            # rounds, which DeepSeek V4 requires in thinking mode. Disable it
            # here instead of failing with a 400 mid-conversation.
            logger.warning("工具调用模式下已强制关闭 thinking")
            thinking = "disabled"
        if thinking != "unset":
            # Raw HTTP uses a top-level field (the OpenAI SDK uses extra_body).
            payload["thinking"] = {"type": thinking}
        return payload

    def _error_from_response(self, status_code: int, body: str) -> LLMError:
        message = body[:500]
        try:
            parsed = json.loads(body)
            message = (
                parsed.get("error", {}).get("message")
                or parsed.get("message")
                or message
            )
        except (json.JSONDecodeError, AttributeError):
            pass
        message = str(message).strip() or "未知错误"
        if status_code == 401:
            message = "DeepSeek API Key 无效或已过期，请检查 .env 中的 LLM_API_KEY"
        elif status_code == 402:
            message = "DeepSeek 账户余额不足，请充值后重试"
        elif status_code == 404:
            message = (
                f"模型 {self.model} 不存在或接口地址错误。请检查 LLM_MODEL / LLM_BASE_URL"
                f"（服务端返回：{message}）"
            )
        elif status_code == 429:
            message = "模型服务限流，请稍后重试"
        elif status_code >= 500:
            message = f"模型服务暂时不可用（HTTP {status_code}）"
        else:
            message = f"模型请求失败（HTTP {status_code}）：{message}"
        return LLMError(
            message,
            status_code=status_code,
            retryable=status_code in _RETRYABLE_STATUS,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        payload = self._payload(messages, tools)
        attempt = 0
        thinking_retry_done = False
        stream_options_retry_done = False

        while True:
            started_streaming = False
            try:
                async with self._get_client().stream(
                    "POST", self.endpoint, json=payload, headers=self._headers()
                ) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", "replace")
                        lowered = body.lower()
                        # Some compatible endpoints reject optional fields. Retry
                        # once without the offending field before failing.
                        if (
                            response.status_code == 400
                            and "thinking" in lowered
                            and "thinking" in payload
                            and not thinking_retry_done
                        ):
                            thinking_retry_done = True
                            payload.pop("thinking", None)
                            logger.warning("服务端不接受 thinking 参数，已自动移除并重试")
                            continue
                        if (
                            response.status_code == 400
                            and "stream_options" in lowered
                            and "stream_options" in payload
                            and not stream_options_retry_done
                        ):
                            stream_options_retry_done = True
                            payload.pop("stream_options", None)
                            logger.warning("服务端不接受 stream_options，已自动移除并重试")
                            continue
                        error = self._error_from_response(response.status_code, body)
                        if error.retryable and attempt < self.settings.llm_max_retries:
                            attempt += 1
                            await asyncio.sleep(min(4.0, 0.5 * (2**attempt)))
                            continue
                        raise error

                    started_streaming = True
                    accumulator = StreamAccumulator()
                    async for event in self._parse_stream(response, accumulator):
                        yield event
                    return
            except _TRANSPORT_ERRORS as exc:
                if not started_streaming and attempt < self.settings.llm_max_retries:
                    attempt += 1
                    logger.warning("连接模型服务失败，第 %s 次重试：%s", attempt, exc)
                    await asyncio.sleep(min(4.0, 0.5 * (2**attempt)))
                    continue
                raise LLMError(f"连接模型服务失败：{exc.__class__.__name__}") from exc
            except LLMError:
                raise
            except Exception as exc:
                logger.exception("模型流式请求异常")
                raise LLMError(f"模型请求异常：{exc.__class__.__name__}") from exc

    async def _parse_stream(
        self, response: httpx.Response, accumulator: StreamAccumulator
    ) -> AsyncIterator[LLMEvent]:
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                logger.debug("忽略无法解析的流式数据：%s", data[:200])
                continue
            if chunk.get("error"):
                message = chunk["error"].get("message") or str(chunk["error"])
                raise LLMError(f"模型返回错误：{message}")
            accumulator.set_usage(chunk.get("usage"))

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                accumulator.text += text
                yield TokenEvent(text)
            reasoning = delta.get("reasoning_content")
            if reasoning:
                # Thinking output is never mixed into the user-visible answer.
                accumulator.reasoning_content += reasoning
            tool_deltas = delta.get("tool_calls")
            if tool_deltas:
                accumulator.add_tool_call_deltas(tool_deltas)
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                accumulator.finish_reason = str(finish_reason)

        if accumulator.usage:
            yield accumulator.usage
        finish_reason = accumulator.finish_reason or (
            "tool_calls" if accumulator.tool_calls() else "stop"
        )
        if finish_reason == "insufficient_system_resource":
            raise LLMError("模型服务资源不足，请稍后重试")
        calls = accumulator.tool_calls()
        if calls:
            yield ToolCallsEvent(calls=calls)
            yield EndEvent(finish_reason="tool_calls")
            return
        if not accumulator.text:
            raise LLMError("模型没有返回任何内容，请重试")
        yield EndEvent(finish_reason=finish_reason)

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None


__all__ = ["DeepSeekClient", "StreamAccumulator"]
