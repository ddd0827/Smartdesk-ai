from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}
MAX_TEXT_CHARS = 500_000


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    text: str
    extension: str


def _title_from_text(text: str, filename: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()[:120]
    stem = Path(filename).stem.strip()
    return stem[:120] or "未命名文档"


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("无法识别文本编码，请使用 UTF-8 或 GBK 编码的文件")


def _extract_pdf(data: bytes, max_pages: int) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf raises several parser-specific errors
        raise DocumentParseError(f"PDF 解析失败：{exc.__class__.__name__}") from exc
    if reader.is_encrypted:
        raise DocumentParseError("暂不支持加密 PDF")
    if len(reader.pages) > max_pages:
        raise DocumentParseError(f"PDF 页数超过限制（最多 {max_pages} 页）")
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages).strip()
    if not text:
        raise DocumentParseError("PDF 中没有可提取的文本（可能是扫描件，请上传文本版）")
    return text


def extract_text(
    filename: str, data: bytes, *, max_pdf_pages: int = 20
) -> ParsedDocument:
    if not data:
        raise DocumentParseError("文件为空")
    safe_name = Path(filename or "upload.txt").name
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentParseError(f"不支持的文件类型 {extension or '(无扩展名)'}，仅支持 {allowed}")

    if extension == ".pdf":
        text = _extract_pdf(data, max_pdf_pages)
    else:
        text = _decode_text(data)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = text.strip()
    if not text:
        raise DocumentParseError("文档没有正文内容")
    if len(text) > MAX_TEXT_CHARS:
        raise DocumentParseError(f"文档正文过长（超过 {MAX_TEXT_CHARS} 字符）")
    return ParsedDocument(
        title=_title_from_text(text, safe_name),
        text=text,
        extension=extension,
    )


__all__ = [
    "DocumentParseError",
    "ParsedDocument",
    "SUPPORTED_EXTENSIONS",
    "extract_text",
]
