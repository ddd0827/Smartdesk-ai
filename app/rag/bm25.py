from __future__ import annotations

import math
from collections import Counter

from app.rag.tokenizer import tokenize, unique_tokens


class BM25Index:
    """Small in-memory BM25 implementation (Okapi, k1=1.5, b=0.75)."""

    def __init__(self, documents: list[tuple[str, str]]) -> None:
        self.ids: list[str] = []
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        document_freq: Counter[str] = Counter()

        for doc_id, text in documents:
            tokens = tokenize(text)
            self.ids.append(doc_id)
            self.term_freqs.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))
            document_freq.update(set(tokens))

        self.n_docs = len(self.ids)
        self.avg_length = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0
        )
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.n_docs or not query.strip() or top_k <= 0:
            return []
        query_terms = unique_tokens(query)
        if not query_terms:
            return []
        k1, b = 1.5, 0.75
        scored: list[tuple[str, float]] = []
        for index, doc_id in enumerate(self.ids):
            freqs = self.term_freqs[index]
            doc_length = self.doc_lengths[index]
            score = 0.0
            for term in query_terms:
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                denominator = freq + k1 * (
                    1 - b + b * doc_length / (self.avg_length or 1.0)
                )
                score += self.idf.get(term, 0.0) * freq * (k1 + 1) / denominator
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]


__all__ = ["BM25Index"]
