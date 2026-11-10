from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TokenDelta:
    text: str


@dataclass(frozen=True, slots=True)
class TraceEvent:
    phase: Literal["start", "end"]
    tool: str
    step: int
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool | None = None
    duration_ms: float | None = None
    summary: str | None = None
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    sources: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class DoneEvent:
    message_id: str
    latency_ms: int
    usage: dict[str, int]
    invalid_citations: list[str]
    interrupted: bool = False
    finish_reason: str = "stop"


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    message: str
    code: str = "agent_error"


AgentEvent = TokenDelta | TraceEvent | SourcesEvent | DoneEvent | ErrorEvent


__all__ = [
    "AgentEvent",
    "DoneEvent",
    "ErrorEvent",
    "SourcesEvent",
    "TokenDelta",
    "TraceEvent",
]
