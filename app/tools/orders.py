from __future__ import annotations

import json
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.store.repo import Repository
from app.tools.base import BaseTool, ToolContext, ToolResult


class LookupOrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(
        min_length=4,
        max_length=32,
        pattern=r"^[A-Za-z0-9-]+$",
        description="订单号，例如 A2026001",
    )


class LookupOrderTool(BaseTool):
    name = "lookup_order"
    description = (
        "查询当前登录客户名下的订单状态、商品和物流信息。"
        "只有用户明确提供了订单号时才调用；不能查询其他客户的订单。"
    )
    args_model = LookupOrderArgs

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    async def run(self, args: LookupOrderArgs, ctx: ToolContext) -> ToolResult:
        order_id = args.order_id.upper()
        order = await self.repo.get_order(order_id)
        # Do not distinguish "missing" from "belongs to someone else" to avoid
        # leaking the existence of another customer's order.
        if not order or order["customer_id"] != ctx.customer_id:
            payload = {
                "ok": False,
                "error": "未找到该订单，或该订单不属于当前账户，无法查询。请核对订单号。",
            }
            return ToolResult(
                ok=False,
                content=json.dumps(payload, ensure_ascii=False),
                summary=f"订单 {order_id} 未找到或无权查看",
                meta={"order_id": order_id, "reason": "not_found_or_forbidden"},
            )

        await self.repo.record_session_order(ctx.session_id, order_id)
        amount = Decimal(order["amount_cents"]) / Decimal(100)
        payload = {
            "ok": True,
            "order": {
                "order_id": order["order_id"],
                "status": order["status"],
                "item_name": order["item_name"],
                "amount_yuan": f"{amount:.2f}",
                "ordered_at": order["ordered_at"],
                "tracking_no": order["tracking_no"],
                "logistics_status": order["logistics_status"],
                "updated_at": order["updated_at"],
            },
        }
        return ToolResult(
            ok=True,
            content=json.dumps(payload, ensure_ascii=False),
            summary=f"订单 {order_id}：{order['status']}，{order['item_name']}",
            meta={
                "order_id": order_id,
                "status": order["status"],
                "item_name": order["item_name"],
            },
        )


__all__ = ["LookupOrderTool"]
