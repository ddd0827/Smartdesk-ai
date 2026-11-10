from __future__ import annotations

import secrets
import sqlite3
import uuid
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.api.deps import get_container
from app.store.repo import utc_now

router = APIRouter(prefix="/mock/itsm", tags=["mock-itsm"])


class MockTicketRequest(BaseModel):
    order_id: str = Field(min_length=4, max_length=32, pattern=r"^[A-Za-z0-9-]+$")
    session_id: str = Field(min_length=4, max_length=64)
    summary: str = Field(min_length=5, max_length=500)
    priority: Literal["low", "normal", "high", "urgent"] = "normal"


def _check_api_key(request: Request, provided: str | None) -> None:
    expected = get_container(request).settings.itsm_api_key_value
    if not expected:
        raise HTTPException(status_code=503, detail="工单系统未配置 API Key")
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="工单系统鉴权失败")


@router.post("/tickets", status_code=201)
async def create_ticket(
    payload: MockTicketRequest,
    request: Request,
    response: Response,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    """Stand-in for a client's ITSM/CRM API.

    Real deployments would replace this endpoint with the customer's system;
    the agent tool only ever talks to a configured base URL.
    """
    # Authenticate before validating request-specific headers/body details so
    # unauthenticated callers cannot probe the endpoint contract.
    _check_api_key(request, x_api_key)
    if not idempotency_key or len(idempotency_key) < 8:
        raise HTTPException(status_code=422, detail="缺少有效的 Idempotency-Key")
    container = get_container(request)
    repo = container.repo

    existing = await repo.get_ticket_by_idempotency(idempotency_key)
    if existing:
        response.status_code = 200
        return {
            "ticket_id": existing["ticket_id"],
            "status": existing["status"],
            "priority": existing["priority"],
            "duplicate": True,
        }

    order = await repo.get_order(payload.order_id.upper())
    if order is None:
        raise HTTPException(status_code=404, detail="订单不存在")
    session = await repo.get_session(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if order["customer_id"] != session["customer_id"]:
        raise HTTPException(status_code=403, detail="订单不属于当前会话客户")

    ticket_id = "T" + uuid.uuid4().hex[:10].upper()
    try:
        ticket = await repo.insert_ticket(
            ticket_id=ticket_id,
            session_id=payload.session_id,
            order_id=order["order_id"],
            summary=payload.summary,
            priority=payload.priority,
            idempotency_key=idempotency_key,
        )
    except sqlite3.IntegrityError:
        existing = await repo.get_ticket_by_idempotency(idempotency_key)
        if existing is None:
            raise
        response.status_code = 200
        return {
            "ticket_id": existing["ticket_id"],
            "status": existing["status"],
            "priority": existing["priority"],
            "duplicate": True,
        }

    return {
        "ticket_id": ticket["ticket_id"],
        "status": ticket["status"],
        "priority": ticket["priority"],
        "duplicate": False,
        "created_at": ticket.get("created_at", utc_now()),
    }


__all__ = ["router"]
