from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.events import (
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    TokenDelta,
    TraceEvent,
)
from app.api.deps import get_container

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=64)


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def _make_title(text: str) -> str:
    first_line = " ".join(text.strip().splitlines()[:1]).strip()
    if not first_line:
        return "新会话"
    return first_line[:24] + ("…" if len(first_line) > 24 else "")


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    container = get_container(request)
    settings = container.settings
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")
    if len(message) > settings.session_max_chars:
        raise HTTPException(
            status_code=400,
            detail=f"消息过长，最多 {settings.session_max_chars} 个字符",
        )

    session: dict[str, Any] | None
    if payload.session_id:
        session = await container.repo.get_session(payload.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        session_id = uuid.uuid4().hex
        session = await container.repo.create_session(
            session_id, _make_title(message), settings.demo_customer_id
        )

    session_id = session["id"]
    lock = container.session_lock(session_id)
    if lock.locked():
        raise HTTPException(status_code=409, detail="该会话正在生成回答，请稍候")
    await lock.acquire()
    lock_acquired = True
    semaphore_acquired = False

    try:
        try:
            await asyncio.wait_for(
                container.stream_semaphore.acquire(), timeout=5.0
            )
            semaphore_acquired = True
        except TimeoutError as exc:
            raise HTTPException(
                status_code=503, detail="当前咨询人数较多，请稍后再试"
            ) from exc

        await container.repo.add_message(session_id, "user", message)
        assistant_message_id = uuid.uuid4().hex
        await container.repo.add_message(
            session_id,
            "assistant",
            "",
            message_id=assistant_message_id,
            meta={"status": "streaming"},
        )
    except BaseException:
        if semaphore_acquired:
            container.stream_semaphore.release()
        if lock_acquired and lock.locked():
            lock.release()
        raise

    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def produce() -> None:
        try:
            async for event in container.agent.run(
                session_id=session_id,
                customer_id=session["customer_id"],
                message_id=assistant_message_id,
                user_text=message,
            ):
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Agent 流式生产任务失败")
            await queue.put(
                ErrorEvent(
                    message="客服系统处理失败，请稍后重试。",
                    code="internal_error",
                )
            )
        finally:
            await queue.put(None)

    producer = asyncio.create_task(produce())
    heartbeat = max(2.0, settings.sse_heartbeat_seconds)

    async def event_stream():
        terminal = False
        try:
            yield _sse(
                "session",
                {
                    "session_id": session_id,
                    "message_id": assistant_message_id,
                    "title": session["title"],
                    "customer_id": session["customer_id"],
                },
            )
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                if event is None:
                    break
                if isinstance(event, TokenDelta):
                    yield _sse("token", {"delta": event.text})
                elif isinstance(event, TraceEvent):
                    yield _sse(
                        "trace",
                        {
                            "phase": event.phase,
                            "tool": event.tool,
                            "step": event.step,
                            "args": event.args,
                            "ok": event.ok,
                            "duration_ms": event.duration_ms,
                            "summary": event.summary,
                            "call_id": event.call_id,
                        },
                    )
                elif isinstance(event, SourcesEvent):
                    yield _sse("sources", {"sources": event.sources})
                elif isinstance(event, DoneEvent):
                    terminal = True
                    yield _sse(
                        "done",
                        {
                            "message_id": event.message_id,
                            "latency_ms": event.latency_ms,
                            "usage": event.usage,
                            "invalid_citations": event.invalid_citations,
                            "finish_reason": event.finish_reason,
                            "interrupted": event.interrupted,
                        },
                    )
                elif isinstance(event, ErrorEvent):
                    terminal = True
                    yield _sse(
                        "error", {"message": event.message, "code": event.code}
                    )
            if not terminal:
                yield _sse(
                    "error",
                    {"message": "生成意外中断，请重试。", "code": "stream_incomplete"},
                )
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer
            if semaphore_acquired:
                container.stream_semaphore.release()
            if lock_acquired and lock.locked():
                lock.release()

    logger.info(
        "chat stream start session=%s chars=%s request_id=%s",
        session_id,
        len(message),
        getattr(request.state, "request_id", "-"),
    )
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


__all__ = ["router"]
