from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app import __version__
from app.api.deps import get_container

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(request: Request, response: Response) -> dict:
    container = get_container(request)
    try:
        database_ok = await container.db.ping()
    except Exception:
        database_ok = False
    if not database_ok:
        response.status_code = 503
    return {
        "status": "ok" if database_ok else "degraded",
        "version": __version__,
        "database": database_ok,
        "index_ready": container.index.ready,
        "indexed_chunks": container.index.size,
        "stale_chunks": container.index.stale_chunks,
        "config": container.settings.safe_summary(),
    }


__all__ = ["router"]
