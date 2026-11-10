from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_BOUNDARIES = "。！？；!?;\n"


class ChunkingError(ValueError):
    pass


@dataclass(frozen=True)
class ChunkDraft:
    seq: int
    section: str
    content: str


@dataclass(frozen=True)
class _Section:
    title: str
    body: str


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sections(text: str) -> list[_Section]:
    """Split Markdown into sections, keeping a heading breadcrumb per section."""
    headings: dict[int, str] = {}
    sections: list[_Section] = []
    buffer: list[str] = []

    def current_title() -> str:
        return " > ".join(headings[level] for level in sorted(headings))

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(_Section(title=current_title(), body=body))
        buffer.clear()

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            for existing in list(headings):
                if existing >= level:
                    del headings[existing]
            headings[level] = match.group(2).strip()
        else:
            buffer.append(line)
    flush()
    if not sections and text.strip():
        sections.append(_Section(title="", body=text.strip()))
    return sections


def _split_body(body: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding window that prefers sentence boundaries and keeps overlap."""
    body = body.strip()
    if not body:
        return []
    chunks: list[str] = []
    start = 0
    length = len(body)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            search_from = start + max(1, int(chunk_size * 0.55))
            cut = -1
            for boundary in _SENTENCE_BOUNDARIES:
                candidate = body.rfind(boundary, search_from, end)
                cut = max(cut, candidate)
            if cut > start:
                end = cut + 1
        piece = body[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        start = next_start
    return chunks


def chunk_text(
    text: str, *, chunk_size: int = 420, overlap: int = 80
) -> list[ChunkDraft]:
    """Chunk a document into sections with a bounded, overlapping window."""
    normalized = _normalize(text)
    if not normalized:
        raise ChunkingError("文档内容为空")
    if overlap >= chunk_size:
        raise ChunkingError("overlap 必须小于 chunk_size")

    drafts: list[ChunkDraft] = []
    for section in _split_sections(normalized):
        for piece in _split_body(section.body, chunk_size, overlap):
            drafts.append(
                ChunkDraft(seq=len(drafts), section=section.title, content=piece)
            )
    if not drafts:
        raise ChunkingError("文档切分后没有可用内容")
    return drafts


__all__ = ["ChunkDraft", "ChunkingError", "chunk_text"]
