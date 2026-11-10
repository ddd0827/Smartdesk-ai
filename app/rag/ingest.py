from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.rag.chunker import ChunkDraft, ChunkingError, chunk_text
from app.rag.index import RetrievalIndex
from app.rag.parsers import DocumentParseError, extract_text
from app.rag.vectors import Embedder, EmbeddingError, pack_vector
from app.store.repo import Repository, utc_now

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc_id: str
    title: str
    filename: str
    chunks: int
    duplicate: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReindexResult:
    documents: int = 0
    chunks: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SeedResult:
    documents_added: int = 0
    documents_skipped: int = 0
    chunks: int = 0
    orders: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def summary(self) -> str:
        return (
            f"知识库导入 {self.documents_added} 篇（跳过 {self.documents_skipped} 篇，"
            f"{self.chunks} 个分块），订单 {self.orders} 条"
        )


class DocumentService:
    def __init__(
        self,
        repo: Repository,
        embedder: Embedder,
        index: RetrievalIndex,
        settings: Settings,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.index = index
        self.settings = settings

    async def ingest_bytes(self, filename: str, data: bytes) -> IngestResult:
        parsed = extract_text(
            filename, data, max_pdf_pages=self.settings.max_pdf_pages
        )
        content_hash = hashlib.sha256(parsed.text.encode("utf-8")).hexdigest()
        existing = await self.repo.find_document_by_hash(content_hash)
        if existing:
            return IngestResult(
                doc_id=existing["id"],
                title=existing["title"],
                filename=existing["filename"],
                chunks=0,
                duplicate=True,
            )

        drafts = chunk_text(
            parsed.text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        vectors = await self.embedder.embed_documents(
            [
                f"{parsed.title}\n{draft.section}\n{draft.content}"
                for draft in drafts
            ]
        )
        if len(vectors) != len(drafts) or not vectors:
            raise EmbeddingError("embedding 返回数量与分块数量不一致")

        doc_id = uuid.uuid4().hex
        created_at = utc_now()
        dim = len(vectors[0])
        chunks = [
            self._chunk_row(doc_id, draft, vector, created_at)
            for draft, vector in zip(drafts, vectors, strict=True)
        ]
        doc = {
            "id": doc_id,
            "title": parsed.title,
            "filename": Path(filename).name,
            "content_hash": content_hash,
            "raw_text": parsed.text,
            "size_bytes": len(data),
            "embedding_provider": self.embedder.name,
            "embedding_model": self.embedder.model,
            "embedding_dim": dim,
            "created_at": created_at,
        }
        try:
            await self.repo.insert_document(doc, chunks)
        except sqlite3.IntegrityError:
            # A concurrent upload of identical content won the race.
            existing = await self.repo.find_document_by_hash(content_hash)
            if existing:
                return IngestResult(
                    doc_id=existing["id"],
                    title=existing["title"],
                    filename=existing["filename"],
                    chunks=0,
                    duplicate=True,
                )
            raise

        await self.index.invalidate()
        logger.info("已导入文档 %s（%s 个分块）", parsed.title, len(chunks))
        return IngestResult(
            doc_id=doc_id,
            title=parsed.title,
            filename=Path(filename).name,
            chunks=len(chunks),
        )

    def _chunk_row(
        self,
        doc_id: str,
        draft: ChunkDraft,
        vector: list[float],
        created_at: str,
    ) -> dict[str, Any]:
        return {
            "id": f"{doc_id}#{draft.seq}",
            "doc_id": doc_id,
            "seq": draft.seq,
            "section": draft.section,
            "content": draft.content,
            "embedding": pack_vector(vector),
            "created_at": created_at,
        }

    async def delete_document(self, doc_id: str) -> bool:
        deleted = await self.repo.delete_document(doc_id)
        if deleted:
            await self.index.invalidate()
        return deleted

    async def reindex_all(self) -> ReindexResult:
        result = ReindexResult()
        documents = await self.repo.list_documents_for_reindex()
        for doc in documents:
            try:
                drafts = chunk_text(
                    doc["raw_text"],
                    chunk_size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                )
                vectors = await self.embedder.embed_documents(
                    [
                        f"{doc['title']}\n{draft.section}\n{draft.content}"
                        for draft in drafts
                    ]
                )
                if len(vectors) != len(drafts) or not vectors:
                    raise EmbeddingError("embedding 返回数量与分块数量不一致")
                now = utc_now()
                chunks = [
                    self._chunk_row(doc["id"], draft, vector, now)
                    for draft, vector in zip(drafts, vectors, strict=True)
                ]
                await self.repo.replace_document_chunks(
                    doc["id"],
                    chunks,
                    self.embedder.name,
                    self.embedder.model,
                    len(vectors[0]),
                )
                result.documents += 1
                result.chunks += len(chunks)
            except (ChunkingError, DocumentParseError, EmbeddingError) as exc:
                result.errors.append(f"{doc['filename']}: {exc}")
                logger.exception("重建索引失败：%s", doc["filename"])
        await self.index.invalidate()
        return result

    async def seed(self, *, force: bool = False) -> SeedResult:
        result = SeedResult()
        docs_exist = await self.repo.count_documents() > 0
        if force or not docs_exist:
            await self._seed_documents(result, force=force)
        orders_exist = await self.repo.count_orders() > 0
        if force or not orders_exist:
            result.orders = await self._seed_orders()
        await self.index.invalidate()
        return result

    async def _seed_documents(self, result: SeedResult, *, force: bool) -> None:
        docs_dir = self.settings.seed_dir / "docs"
        if not docs_dir.exists():
            logger.warning("未找到种子知识库目录：%s", docs_dir)
            return
        for path in sorted(docs_dir.glob("*.md")):
            out = await self.ingest_bytes(path.name, path.read_bytes())
            if out.duplicate:
                result.documents_skipped += 1
                if force:
                    continue
            else:
                result.documents_added += 1
                result.chunks += out.chunks
        logger.info("知识库种子数据：%s", result.summary)

    async def _seed_orders(self) -> int:
        orders_path = self.settings.seed_dir / "orders.json"
        if not orders_path.exists():
            logger.warning("未找到种子订单文件：%s", orders_path)
            return 0
        raw = json.loads(orders_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            logger.error("种子订单格式错误：期望 JSON 数组")
            return 0
        valid: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            required = {"order_id", "customer_id", "customer_name", "status", "item_name"}
            if not required.issubset(item):
                logger.warning("跳过缺少字段的订单：%s", item.get("order_id", "?"))
                continue
            valid.append(
                {
                    "order_id": str(item["order_id"]),
                    "customer_id": str(item["customer_id"]),
                    "customer_name": str(item["customer_name"]),
                    "status": str(item["status"]),
                    "item_name": str(item["item_name"]),
                    "amount_cents": int(item.get("amount_cents", 0)),
                    "ordered_at": str(item.get("ordered_at", "")),
                    "tracking_no": str(item.get("tracking_no", "")),
                    "logistics_status": str(item.get("logistics_status", "")),
                    "updated_at": str(item.get("updated_at", utc_now())),
                }
            )
        return await self.repo.upsert_orders(valid)


__all__ = [
    "DocumentService",
    "IngestResult",
    "ReindexResult",
    "SeedResult",
]
