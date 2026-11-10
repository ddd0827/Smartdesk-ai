from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import get_container

router = APIRouter(tags=["system"])


@router.get("/stats")
async def stats(request: Request) -> dict:
    container = get_container(request)
    data = await container.repo.stats(
        container.embedder.name, container.embedder.model, container.embedder.dim
    )
    return {
        **data,
        "mode": "mock" if container.settings.is_mock_llm else "live",
        "model": container.llm.model,
        "embedding": container.embedder.signature,
        "index_ready": container.index.ready,
    }


__all__ = ["router"]
