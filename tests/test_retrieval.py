from __future__ import annotations

import numpy as np

from app.rag.bm25 import BM25Index
from app.rag.vectors import (
    HashNgramVectorizer,
    normalize_rows,
    pack_vector,
    unpack_vector,
)


def test_bm25_orders_by_relevance() -> None:
    index = BM25Index(
        [
            ("a", "退货政策 七天无理由 运费"),
            ("b", "发票 退款 银行卡 到账"),
        ]
    )
    results = index.search("发票退款", 2)
    assert results
    assert results[0][0] == "b"
    if len(results) > 1:
        assert results[0][1] > results[1][1]


def test_hash_vector_is_deterministic_and_normalized() -> None:
    first = HashNgramVectorizer(128)
    second = HashNgramVectorizer(128)
    vector_a = first._embed_one("退货 政策 七天无理由")
    vector_b = second._embed_one("退货 政策 七天无理由")
    assert vector_a == vector_b
    assert np.isclose(np.linalg.norm(vector_a), 1.0)
    packed = pack_vector(vector_a)
    unpacked = unpack_vector(packed)
    assert np.allclose(unpacked, np.asarray(vector_a, dtype=np.float32))


def test_normalize_rows_handles_zero_vectors() -> None:
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    normalized = normalize_rows(matrix)
    assert np.allclose(normalized[0], [0.0, 0.0])
    assert np.isclose(np.linalg.norm(normalized[1]), 1.0)


async def test_seeded_hybrid_retrieval(seeded) -> None:
    results = await seeded.index.search_hybrid("七天无理由退货", 3)
    assert results
    assert results[0].filename == "01-退换货政策.md"
    assert results[0].score > 0
    assert results[0].snippet
    assert len(results[0].snippet) <= 221
    source = results[0].to_source("S1")
    assert source["id"] == "S1"
    assert source["title"]


async def test_hybrid_exposes_rank_metadata(seeded) -> None:
    results = await seeded.index.search_hybrid("退货 物流", 5)
    assert any(
        item.bm25_rank is not None and item.vector_rank is not None
        for item in results
    )


async def test_empty_queries_return_nothing(seeded) -> None:
    assert await seeded.index.search_bm25("", 3) == []
    assert await seeded.index.search_vector("   ", 3) == []
    assert await seeded.index.search_hybrid("", 3) == []


async def test_ingest_invalidates_and_indexes_new_document(seeded) -> None:
    content = "# 量子保修说明\n\n## 保修期限\n\n量子系列产品保修 99 年。".encode()
    result = await seeded.service.ingest_bytes("99-量子保修.md", content)
    assert result.duplicate is False
    assert result.chunks >= 1
    results = await seeded.index.search_hybrid("量子系列保修多少年", 3)
    assert results
    assert any(item.filename == "99-量子保修.md" for item in results)

    duplicate = await seeded.service.ingest_bytes("99-量子保修.md", content)
    assert duplicate.duplicate is True
    assert duplicate.doc_id == result.doc_id
