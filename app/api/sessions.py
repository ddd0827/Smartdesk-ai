from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.deps import get_container

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("")
async def list_sessions(request: Request) -> dict:
    container = get_container(request)
    sessions = await container.repo.list_sessions()
    return {"sessions": sessions}


@router.get("/{session_id}/messages")
async def session_messages(session_id: str, request: Request) -> dict:
    container = get_container(request)
    session = await container.repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = await container.repo.list_messages(session_id)
    return {
        "session": session,
        "messages": [
            {
                "id": message["id"],
                "role": message["role"],
                "content": message["content"],
                "meta": message["meta"],
                "feedback": message["feedback"],
                "created_at": message["created_at"],
            }
            for message in messages
        ],
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    container = get_container(request)
    deleted = await container.repo.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    lock = container.session_locks.get(session_id)
    if lock is not None and not lock.locked():
        container.session_locks.pop(session_id, None)
    return {"deleted": True, "session_id": session_id}


__all__ = ["router"]
