from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.deps import get_container, require_admin
from app.rag.chunker import ChunkingError
from app.rag.parsers import SUPPORTED_EXTENSIONS, DocumentParseError
from app.rag.vectors import EmbeddingError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents(request: Request) -> dict:
    container = get_container(request)
    documents = await container.repo.list_documents()
    return {
        "documents": documents,
        "count": len(documents),
        "chunks": sum(int(item["chunk_count"]) for item in documents),
    }


@router.post("", dependencies=[Depends(require_admin)])
async def upload_document(
    request: Request, file: UploadFile = File(...)
) -> dict:
    container = get_container(request)
    settings = container.settings
    filename = Path(file.filename or "upload.txt").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型，仅支持 {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    await file.close()
    if len(data) > limit:
        raise HTTPException(
            status_code=413, detail=f"文件超过 {settings.max_upload_mb} MB 限制"
        )
    try:
        result = await container.documents.ingest_bytes(filename, data)
    except (DocumentParseError, ChunkingError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        **result.as_dict(),
        "message": (
            f"文档已存在：{result.title}" if result.duplicate else "文档已入库"
        ),
    }


@router.post("/reindex", dependencies=[Depends(require_admin)])
async def reindex_documents(request: Request) -> dict:
    container = get_container(request)
    try:
        result = await container.documents.reindex_all()
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        **result.as_dict(),
        "message": (
            "重建索引完成" if not result.errors else "重建索引完成，但有部分失败"
        ),
    }


@router.delete("/{doc_id}", dependencies=[Depends(require_admin)])
async def delete_document(doc_id: str, request: Request) -> dict:
    container = get_container(request)
    deleted = await container.documents.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"deleted": True, "doc_id": doc_id}


__all__ = ["router"]
