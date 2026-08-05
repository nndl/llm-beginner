"""PDF text normalization and character-based chunking."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Extract readable text from a PDF without using source documents."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping character chunks, preferring paragraph breaks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            window = text[start:end]
            break_at = max(window.rfind("\n\n"), window.rfind("。"), window.rfind("！"), window.rfind("？"))
            if break_at >= chunk_size // 2:
                end = start + break_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
    return chunks