from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.store.repo import Repository
from app.tools.base import BaseTool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class CreateTicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(
        min_length=4,
        max_length=32,
        pattern=r"^[A-Za-z0-9-]+$",
        description="需要跟进的订单号",
    )
    summary: str = Field(
        min_length=5, max_length=500, description="问题摘要，包含用户诉求"
    )
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        default="normal", description="优先级"
    )


class CreateTicketTool(BaseTool):
    name = "create_ticket"
    description = (
        "在当前会话中创建售后工单。必须先调用 lookup_order 查询到该订单，"
        "并且用户明确要求人工跟进/建工单后才能调用；同一会话最多创建 3 张工单。"
    )
    args_model = CreateTicketArgs

    def __init__(
        self,
        repo: Repository,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 10.0,
        max_tickets_per_session: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.repo = repo
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self.max_tickets_per_session = max_tickets_per_session
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=False,
                transport=self._transport,
            )
        return self._client

    @staticmethod
    def _idempotency_key(
        session_id: str, order_id: str, summary: str, priority: str
    ) -> str:
        raw = f"{session_id}|{order_id}|{summary}|{priority}".encode()
        return hashlib.sha256(raw).hexdigest()

    async def run(self, args: CreateTicketArgs, ctx: ToolContext) -> ToolResult:
        order_id = args.order_id.upper()
        if not await self.repo.has_session_order(ctx.session_id, order_id):
            return self._error(
                "创建工单前必须先查询该订单，且订单必须属于当前客户。", "order_not_checked"
            )
        count = await self.repo.count_session_tickets(ctx.session_id)
        if count >= self.max_tickets_per_session:
            return self._error(
                f"当前会话已创建 {count} 张工单，达到上限（{self.max_tickets_per_session} 张）。",
                "ticket_limit",
            )

        key = self._idempotency_key(
            ctx.session_id, order_id, args.summary.strip(), args.priority
        )
        existing = await self.repo.get_ticket_by_idempotency(key)
        if existing:
            return self._success(
                ticket_id=existing["ticket_id"],
                status=existing["status"],
                priority=existing["priority"],
                duplicate=True,
            )

        payload = {
            "order_id": order_id,
            "session_id": ctx.session_id,
            "summary": args.summary.strip(),
            "priority": args.priority,
        }
        headers = {
            "X-API-Key": self._api_key,
            "Idempotency-Key": key,
            "Content-Type": "application/json",
        }
        last_error: str | None = None
        for attempt in range(3):
            try:
                response = await self._get_client().post(
                    f"{self.base_url}/api/mock/itsm/tickets",
                    json=payload,
                    headers=headers,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_error = exc.__class__.__name__
                await asyncio.sleep(0.4 * (2**attempt))
                continue

            if response.status_code in (200, 201):
                data = response.json()
                return self._success(
                    ticket_id=data["ticket_id"],
                    status=data.get("status", "open"),
                    priority=data.get("priority", args.priority),
                    duplicate=bool(data.get("duplicate")),
                )
            if response.status_code == 401:
                return self._error(
                    "工单系统鉴权失败，请联系管理员检查 ITSM_API_KEY。", "itsm_auth"
                )
            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                await asyncio.sleep(0.4 * (2**attempt))
                continue
            message = response.text[:200]
            logger.warning("ITSM 创建工单失败：HTTP %s %s", response.status_code, message)
            return self._error(
                f"工单系统拒绝了请求（HTTP {response.status_code}），请稍后再试。",
                "itsm_rejected",
            )

        logger.error("ITSM 创建工单重试后仍失败：%s", last_error)
        return self._error(
            "工单系统暂时不可用，请稍后重试，或直接联系人工客服。", "itsm_unavailable"
        )

    def _success(
        self, *, ticket_id: str, status: str, priority: str, duplicate: bool
    ) -> ToolResult:
        payload = {
            "ok": True,
            "ticket": {
                "ticket_id": ticket_id,
                "status": status,
                "priority": priority,
            },
            "message": "工单已创建，客服会在 24 小时内联系用户。",
            "duplicate": duplicate,
        }
        return ToolResult(
            ok=True,
            content=json.dumps(payload, ensure_ascii=False),
            summary=(
                f"工单 {ticket_id} 已创建（优先级 {priority}）"
                if not duplicate
                else f"重复请求，返回已有工单 {ticket_id}"
            ),
            meta={
                "ticket_id": ticket_id,
                "status": status,
                "priority": priority,
                "duplicate": duplicate,
            },
        )

    def _error(self, message: str, reason: str) -> ToolResult:
        return ToolResult(
            ok=False,
            content=json.dumps({"ok": False, "error": message}, ensure_ascii=False),
            summary=message,
            meta={"reason": reason},
        )

    async def set_transport(
        self, transport: httpx.AsyncBaseTransport | None
    ) -> None:
        """Testing hook: route ITSM calls through an ASGI/mock transport."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._transport = transport

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["CreateTicketTool"]
