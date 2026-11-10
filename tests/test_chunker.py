from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from app.rag.chunker import ChunkingError, chunk_text
from app.rag.parsers import DocumentParseError, extract_text


def test_chunks_respect_size_and_keep_sections() -> None:
    text = "# 售后手册\n\n## 退货政策\n\n" + "这是一段关于退货政策的说明。" * 60
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 300 for chunk in chunks)
    assert chunks[0].section.startswith("售后手册 > 退货政策")
    assert all(chunk.seq == index for index, chunk in enumerate(chunks))


def test_empty_document_is_rejected() -> None:
    with pytest.raises(ChunkingError):
        chunk_text("   \n\n  ")


def test_long_unpunctuated_text_is_hard_split() -> None:
    chunks = chunk_text("测" * 1000, chunk_size=200, overlap=20)
    assert len(chunks) >= 5
    assert all(len(chunk.content) <= 200 for chunk in chunks)


def test_overlap_parameter_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ChunkingError):
        chunk_text("正文" * 200, chunk_size=100, overlap=100)


def test_markdown_title_and_gbk_decoding() -> None:
    raw = "# 发票与退款\n\n## 退款时效\n\n银行卡退款一般 3-7 个工作日到账。".encode("gbk")
    parsed = extract_text("发票.txt", raw)
    assert parsed.title == "发票与退款"
    assert "3-7 个工作日" in parsed.text
    assert parsed.extension == ".txt"


def test_parser_rejects_unsupported_extension() -> None:
    with pytest.raises(DocumentParseError, match="不支持"):
        extract_text("virus.exe", b"data")


def test_parser_rejects_empty_pdf_text() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(DocumentParseError, match="没有可提取的文本"):
        extract_text("blank.pdf", buffer.getvalue())


def test_parser_rejects_pdf_over_page_limit() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(DocumentParseError, match="页数"):
        extract_text("two.pdf", buffer.getvalue(), max_pdf_pages=1)
