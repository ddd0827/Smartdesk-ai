from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np

from app.config import Settings
from app.rag.tokenizer import tokenize


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str
    model: str
    dim: int

    @property
    def signature(self) -> str: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def close(self) -> None: ...


def pack_vector(vector: Sequence[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashNgramVectorizer:
    """Deterministic signed feature hashing over Chinese/ASCII n-grams.

    This is a lexical vectorizer, not a semantic embedding model. It exists so
    ingestion, hybrid retrieval and tests run offline with zero extra keys.
    The fixed blake2b seed makes vectors reproducible across processes (never
    use Python's randomized ``hash()`` here).
    """

    name = "hash"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.model = f"hash-ngram-{dim}"

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.model}"

    def _embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(
                token.encode("utf-8"), digest_size=8, person=b"smartdesk"
            ).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    async def close(self) -> None:
        return None


class OpenAICompatibleEmbedder:
    """Embedding client for OpenAI-compatible endpoints (e.g. SiliconFlow BGE-M3)."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._timeout = timeout
        self.dim = 0
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.model}"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers=headers,
                follow_redirects=False,
            )
        return self._client

    async def _request_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "encoding_format": "float"}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await self._get_client().post(
                    f"{self.base_url}/embeddings", json=payload
                )
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = EmbeddingError(
                        f"embedding 服务暂时不可用（HTTP {response.status_code}）"
                    )
                    await asyncio.sleep(0.4 * (2**attempt))
                    continue
                if response.status_code != 200:
                    message = response.text[:300]
                    raise EmbeddingError(
                        f"embedding 请求失败（HTTP {response.status_code}）：{message}"
                    )
                data = response.json().get("data", [])
                data = sorted(data, key=lambda item: item.get("index", 0))
                vectors = [item["embedding"] for item in data]
                if len(vectors) != len(texts):
                    raise EmbeddingError("embedding 返回数量与请求不一致")
                for vector in vectors:
                    if not isinstance(vector, list) or not vector:
                        raise EmbeddingError("embedding 返回了空向量")
                return vectors
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                last_error = exc
                await asyncio.sleep(0.4 * (2**attempt))
        raise EmbeddingError(f"embedding 服务连接失败：{last_error}")

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), 16):
            batch = list(texts[start : start + 16])
            vectors = await self._request_batch(batch)
            if self.dim == 0:
                async with self._lock:
                    if self.dim == 0:
                        self.dim = len(vectors[0])
            results.extend(vectors)
        return results

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._request_batch([text])
        if self.dim == 0:
            self.dim = len(vectors[0])
        return vectors[0]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hash":
        return HashNgramVectorizer(dim=settings.embedding_dim)
    if not settings.embedding_api_key_value:
        raise EmbeddingError("EMBEDDING_API_KEY 为空，无法使用远程 embedding")
    return OpenAICompatibleEmbedder(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key_value,
        model=settings.embedding_model,
        timeout=settings.embedding_timeout_seconds,
    )


__all__ = [
    "Embedder",
    "EmbeddingError",
    "HashNgramVectorizer",
    "OpenAICompatibleEmbedder",
    "build_embedder",
    "normalize_rows",
    "pack_vector",
    "unpack_vector",
]
