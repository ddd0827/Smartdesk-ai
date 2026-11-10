from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from app.rag.index import RetrievedChunk


class ToolError(RuntimeError):
    pass


@dataclass
class SourceRegistry:
    """Assigns stable ``S1..Sn`` labels to chunks retrieved in one turn."""

    _sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    _next: int = 1

    def add(self, chunk: RetrievedChunk) -> str:
        for source_id, source in self._sources.items():
            if source["chunk_id"] == chunk.chunk_id:
                return source_id
        source_id = f"S{self._next}"
        self._next += 1
        self._sources[source_id] = chunk.to_source(source_id)
        return source_id

    def add_many(self, chunks: list[RetrievedChunk]) -> list[str]:
        return [self.add(chunk) for chunk in chunks]

    def snapshot(self) -> list[dict[str, Any]]:
        def sort_key(item: tuple[str, dict[str, Any]]) -> int:
            try:
                return int(item[0][1:])
            except ValueError:
                return 10**9

        return [dict(value) for _, value in sorted(self._sources.items(), key=sort_key)]

    def __len__(self) -> int:
        return len(self._sources)


@dataclass
class ToolContext:
    session_id: str
    customer_id: str
    sources: SourceRegistry


@dataclass
class ToolResult:
    ok: bool
    content: str
    summary: str
    meta: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str
    description: str
    args_model: type[BaseModel]

    async def run(self, args: Any, ctx: ToolContext) -> ToolResult: ...

    def schema(self) -> dict[str, Any]: ...


class BaseTool:
    name: str = ""
    description: str = ""
    args_model: type[BaseModel]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


__all__ = [
    "BaseTool",
    "SourceRegistry",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
]
