from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from pydantic import ValidationError

from app.tools.base import Tool, ToolContext, ToolError, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Validated, timeout-guarded execution for a fixed allowlist of tools."""

    def __init__(
        self,
        tools: list[Tool],
        *,
        max_result_chars: int = 4000,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self.max_result_chars = max_result_chars
        self.timeout_seconds = timeout_seconds

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def _error(self, message: str, *, reason: str) -> ToolResult:
        return ToolResult(
            ok=False,
            content=json.dumps({"ok": False, "error": message}, ensure_ascii=False),
            summary=message,
            meta={"reason": reason},
        )

    async def execute(
        self, name: str, raw_arguments: str, ctx: ToolContext
    ) -> tuple[ToolResult, float]:
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return self._error(f"未知工具：{name}", reason="unknown_tool"), 0.0

        try:
            raw = json.loads(raw_arguments or "{}")
            if not isinstance(raw, dict):
                raise ValueError("参数必须是 JSON 对象")
            args = tool.args_model.model_validate(raw)
        except json.JSONDecodeError:
            return self._error(
                "工具参数不是合法 JSON，请重新生成正确的参数。", reason="bad_json"
            ), (time.perf_counter() - started) * 1000
        except (ValidationError, ValueError) as exc:
            return self._error(
                f"工具参数校验失败：{self._format_validation_error(exc)}",
                reason="validation",
            ), (time.perf_counter() - started) * 1000

        try:
            result = await asyncio.wait_for(
                tool.run(args, ctx), timeout=self.timeout_seconds
            )
        except TimeoutError:
            logger.warning("工具 %s 执行超时", name)
            result = self._error(
                f"工具 {name} 执行超时，请稍后重试。", reason="timeout"
            )
        except ToolError as exc:
            result = self._error(str(exc), reason="tool_error")
        except Exception:
            logger.exception("工具 %s 执行失败", name)
            result = self._error(
                f"工具 {name} 执行失败，请稍后重试。", reason="internal"
            )

        duration_ms = (time.perf_counter() - started) * 1000
        if len(result.content) > self.max_result_chars:
            result = ToolResult(
                ok=result.ok,
                content=(
                    result.content[: self.max_result_chars]
                    + "\n…（工具结果过长，已截断）"
                ),
                summary=result.summary,
                meta=result.meta,
            )
        logger.info(
            "工具调用 name=%s ok=%s duration_ms=%.0f summary=%s",
            name,
            result.ok,
            duration_ms,
            result.summary[:80],
        )
        return result, duration_ms

    @staticmethod
    def _format_validation_error(exc: Exception) -> str:
        if isinstance(exc, ValidationError):
            details = []
            for error in exc.errors()[:3]:
                location = ".".join(str(part) for part in error.get("loc", ()))
                details.append(f"{location}: {error.get('msg', '')}")
            return "；".join(details)
        return str(exc)

    async def close(self) -> None:
        for tool in self._tools.values():
            close = getattr(tool, "close", None)
            if close is not None:
                await close()


__all__ = ["ToolRegistry"]
