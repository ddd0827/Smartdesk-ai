from __future__ import annotations

import re

_CJK_RE = re.compile(r"[㐀-䶿一-鿿]+")
_ASCII_RE = re.compile(r"[a-z0-9][a-z0-9_.+-]*")


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text without external dictionaries.

    Chinese runs produce unigrams and bigrams, which gives BM25 and the hashed
    vectorizer useful lexical signal without a segmentation dependency. ASCII
    words/numbers are lower-cased and kept whole.
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    for match in _ASCII_RE.finditer(lowered):
        tokens.append(match.group(0))
    for match in _CJK_RE.finditer(lowered):
        run = match.group(0)
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def unique_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(tokenize(text)))


__all__ = ["tokenize", "unique_tokens"]
