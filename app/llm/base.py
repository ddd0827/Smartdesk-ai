from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class LLMError(RuntimeError):
    def __init__(
        self, message: str, *, status_code: int | None = None, retryable: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ToolArgumentsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolArgumentsError(
                f"工具 {self.name} 的参数不是合法 JSON：{self.arguments[:200]}"
            ) from exc
        if not isinstance(value, dict):
            raise ToolArgumentsError(f"工具 {self.name} 的参数必须是 JSON 对象")
        return value


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallsEvent:
    calls: list[ToolCall]


@dataclass(frozen=True, slots=True)
class UsageEvent:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class EndEvent:
    finish_reason: str


LLMEvent = TokenEvent | ToolCallsEvent | UsageEvent | EndEvent


@runtime_checkable
class LLMClient(Protocol):
    model: str

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]: ...

    async def close(self) -> None: ...


__all__ = [
    "EndEvent",
    "LLMClient",
    "LLMEvent",
    "LLMError",
    "TokenEvent",
    "ToolArgumentsError",
    "ToolCall",
    "ToolCallsEvent",
    "UsageEvent",
]
