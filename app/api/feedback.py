from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.deps import get_container

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    message_id: str = Field(min_length=8, max_length=64)
    rating: Literal[-1, 1]
    comment: str = Field(default="", max_length=500)


@router.post("/feedback")
async def submit_feedback(payload: FeedbackRequest, request: Request) -> dict:
    container = get_container(request)
    message = await container.repo.get_message(payload.message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="消息不存在")
    if message["role"] != "assistant":
        raise HTTPException(status_code=400, detail="只能评价助手消息")
    feedback = await container.repo.upsert_feedback(
        payload.message_id, payload.rating, payload.comment.strip()
    )
    return {
        "ok": True,
        "feedback": {
            "rating": feedback.get("rating"),
            "comment": feedback.get("comment", ""),
        },
    }


__all__ = ["router"]
