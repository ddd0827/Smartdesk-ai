from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import Settings
from app.rag.bm25 import BM25Index
from app.rag.vectors import Embedder, EmbeddingError, normalize_rows, unpack_vector
from app.store.repo import Repository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexedChunk:
    id: str
    doc_id: str
    seq: int
    title: str
    filename: str
    section: str
    content: str
    embedding: np.ndarray


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    seq: int
    title: str
    filename: str
    section: str
    content: str
    score: float
    bm25_score: float | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None

    @property
    def snippet(self) -> str:
        text = " ".join(self.content.split())
        return text if len(text) <= 220 else f"{text[:220]}…"

    def to_source(self, source_id: str) -> dict[str, Any]:
        return {
            "id": source_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "filename": self.filename,
            "section": self.section,
            "snippet": self.snippet,
            "score": round(float(self.score), 5),
        }


class RetrievalIndex:
    """Single-process hybrid index over SQLite-persisted chunks.

    The index is rebuilt lazily after any document mutation. BM25 and vector
    scores are fused with Reciprocal Rank Fusion. For a production corpus this
    would move to a vector database, but a few hundred chunks are comfortably
    handled in memory and the trade-off is documented.
    """

    def __init__(self, repo: Repository, embedder: Embedder, settings: Settings) -> None:
        self.repo = repo
        self.embedder = embedder
        self.settings = settings
        self._chunks: list[IndexedChunk] = []
        self._by_id: dict[str, IndexedChunk] = {}
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._bm25 = BM25Index([])
        self._ready = False
        self._stale_chunks = 0
        self._lock = asyncio.Lock()
        self._last_rebuild_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def stale_chunks(self) -> int:
        return self._stale_chunks

    async def invalidate(self) -> None:
        self._ready = False

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            await self._rebuild()

    async def _rebuild(self) -> None:
        try:
            rows = await self.repo.list_index_rows()
        except Exception as exc:  # pragma: no cover - surfaced through logs/health
            self._last_rebuild_error = str(exc)
            raise

        valid: list[IndexedChunk] = []
        vectors: list[np.ndarray] = []
        stale = 0
        for row in rows:
            if (
                row["embedding_provider"] != self.embedder.name
                or row["embedding_model"] != self.embedder.model
            ):
                stale += 1
                continue
            vector = unpack_vector(row["embedding"]).copy()
            if vector.size == 0 or (self.embedder.dim and vector.size != self.embedder.dim):
                stale += 1
                continue
            valid.append(
                IndexedChunk(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    seq=row["seq"],
                    title=row["title"],
                    filename=row["filename"],
                    section=row["section"],
                    content=row["content"],
                    embedding=vector,
                )
            )
            vectors.append(vector)

        # Guard against mixing dimensions if a remote model changed size.
        if vectors:
            dims = Counter(v.shape[0] for v in vectors)
            common_dim, _ = dims.most_common(1)[0]
            if len(dims) > 1:
                stale += sum(
                    count for dim, count in dims.items() if dim != common_dim
                )
                keep = [i for i, v in enumerate(vectors) if v.shape[0] == common_dim]
                valid = [valid[i] for i in keep]
                vectors = [vectors[i] for i in keep]

        self._chunks = valid
        self._by_id = {chunk.id: chunk for chunk in valid}
        self._matrix = (
            normalize_rows(np.vstack(vectors).astype(np.float32))
            if vectors
            else np.empty((0, 0), dtype=np.float32)
        )
        self._bm25 = BM25Index(
            [
                (chunk.id, f"{chunk.title} {chunk.section} {chunk.content}")
                for chunk in valid
            ]
        )
        self._stale_chunks = stale
        self._last_rebuild_error = None
        self._ready = True
        if stale:
            logger.warning(
                "检索索引忽略了 %s 个使用旧 embedding 配置的分块，请重建索引", stale
            )
        logger.info("检索索引就绪：%s 个分块", len(valid))

    def _to_retrieved(
        self,
        chunk_id: str,
        score: float,
        *,
        bm25_score: float | None = None,
        vector_score: float | None = None,
        bm25_rank: int | None = None,
        vector_rank: int | None = None,
    ) -> RetrievedChunk:
        chunk = self._by_id[chunk_id]
        return RetrievedChunk(
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            seq=chunk.seq,
            title=chunk.title,
            filename=chunk.filename,
            section=chunk.section,
            content=chunk.content,
            score=score,
            bm25_score=bm25_score,
            vector_score=vector_score,
            bm25_rank=bm25_rank,
            vector_rank=vector_rank,
        )

    async def search_bm25(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        await self.ensure_ready()
        limit = top_k or self.settings.retrieval_top_k
        results = self._bm25.search(query, limit)
        return [
            self._to_retrieved(
                chunk_id, score, bm25_score=score, bm25_rank=rank
            )
            for rank, (chunk_id, score) in enumerate(results, start=1)
        ]

    async def search_vector(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        await self.ensure_ready()
        limit = top_k or self.settings.retrieval_top_k
        if not self._chunks or self._matrix.size == 0:
            return []
        query_vector = np.asarray(
            await self.embedder.embed_query(query), dtype=np.float32
        )
        if query_vector.size != self._matrix.shape[1]:
            raise EmbeddingError(
                "查询向量维度与索引不一致，请重建索引（POST /api/documents/reindex）"
            )
        norm = float(np.linalg.norm(query_vector))
        if norm == 0:
            return []
        scores = self._matrix @ (query_vector / norm)
        ranked = sorted(
            enumerate(scores.tolist()),
            key=lambda item: (-item[1], self._chunks[item[0]].id),
        )
        results: list[RetrievedChunk] = []
        for index, score in ranked:
            if score <= 0:
                continue
            chunk = self._chunks[index]
            results.append(
                self._to_retrieved(
                    chunk.id,
                    score,
                    vector_score=score,
                    vector_rank=len(results) + 1,
                )
            )
            if len(results) >= limit:
                break
        return results

    async def search_hybrid(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        await self.ensure_ready()
        limit = top_k or self.settings.retrieval_top_k
        candidates = max(self.settings.retrieval_candidates, limit * 3)

        bm25_results = await self.search_bm25(query, candidates)
        vector_results = await self.search_vector(query, candidates)

        rrf_scores: dict[str, float] = {}
        bm25_meta: dict[str, tuple[float, int]] = {}
        vector_meta: dict[str, tuple[float, int]] = {}
        for rank, item in enumerate(bm25_results, start=1):
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + 1.0 / (
                self.settings.rrf_k + rank
            )
            bm25_meta[item.chunk_id] = (item.bm25_score or 0.0, rank)
        for rank, item in enumerate(vector_results, start=1):
            rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + 1.0 / (
                self.settings.rrf_k + rank
            )
            vector_meta[item.chunk_id] = (item.vector_score or 0.0, rank)

        merged: list[RetrievedChunk] = []
        for chunk_id, rrf_score in rrf_scores.items():
            bm25_score, bm25_rank = bm25_meta.get(chunk_id, (None, None))  # type: ignore[assignment]
            vector_score, vector_rank = vector_meta.get(chunk_id, (None, None))  # type: ignore[assignment]
            merged.append(
                self._to_retrieved(
                    chunk_id,
                    rrf_score,
                    bm25_score=bm25_score,
                    vector_score=vector_score,
                    bm25_rank=bm25_rank,
                    vector_rank=vector_rank,
                )
            )
        merged.sort(
            key=lambda item: (
                -item.score,
                min(
                    item.bm25_rank or 10**6,
                    item.vector_rank or 10**6,
                ),
                item.chunk_id,
            )
        )
        return merged[:limit]

    async def health(self) -> dict[str, Any]:
        await self.ensure_ready()
        return {
            "chunks_indexed": len(self._chunks),
            "stale_chunks": self._stale_chunks,
            "embedding": self.embedder.signature,
            "last_error": self._last_rebuild_error,
        }


__all__ = ["IndexedChunk", "RetrievalIndex", "RetrievedChunk"]
