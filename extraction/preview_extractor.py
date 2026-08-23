import re
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
from extractors.registry import extract_text_from_file
import config


def extract_preview(filepath: str, keyword: str, page_hint: int = None, snippet: str = "") -> str:
    """Extract contextual preview around keyword, preferring snippet if provided."""
    # If snippet is long enough and contains keyword, use it directly
    if snippet and len(snippet) >= 100 and keyword.lower() in snippet.lower():
        return snippet

    path = Path(filepath)
    if not path.exists():
        return ""

    try:
        if path.suffix.lower() == ".pdf":
            return extract_pdf_preview(path, keyword, page_hint)
        else:
            result = extract_text_from_file(path)
            text = result.get("text", "")
            return window_around_keyword(text, keyword, config.PREVIEW_CHAR_WINDOW)
    except Exception as e:
        print(f"Preview extraction error for {filepath}: {e}")
        return ""


def extract_pdf_preview(pdf_path: Path, keyword: str, page_hint: int = None, window_chars: int = None) -> str:
    """Multi-page PDF preview around keyword occurrence."""
    if window_chars is None:
        window_chars = config.PREVIEW_CHAR_WINDOW

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return ""

    page_count = doc.page_count
    if page_count == 0:
        return ""

    # Find the best start page (0-indexed)
    start_page = None
    if page_hint is not None and 0 < page_hint <= page_count:
        start_page = page_hint - 1
    else:
        candidate_pages = []
        for i in range(page_count):
            text = doc[i].get_text()
            if keyword.lower() in text.lower():
                candidate_pages.append((i, len(text)))
        if candidate_pages:
            # Prefer page with more text (skip sparse title pages)
            # Simple approach: choose the page with maximum text length
            candidate_pages.sort(key=lambda x: x[1], reverse=True)
            start_page = candidate_pages[0][0]
        else:
            start_page = 0

    # Collect up to 5 pages around start_page
    pages = [start_page]
    offset = 1
    max_pages = 3
    while len(pages) < max_pages:
        added = False
        prev = start_page - offset
        if prev >= 0 and prev not in pages:
            pages.append(prev)
            added = True
        nxt = start_page + offset
        if nxt < page_count and nxt not in pages:
            pages.append(nxt)
            added = True
        if not added:
            break
        combined = "\n".join(doc[p].get_text() for p in sorted(pages))
        if len(combined) >= window_chars:
            break
        offset += 1

    pages = sorted(set(pages))
    combined = "\n".join(doc[p].get_text() for p in pages)
    return window_around_keyword(combined, keyword, window_chars)


def window_around_keyword(text: str, keyword: str, window_chars: int) -> str:
    """Return substring centered around first occurrence of keyword."""
    if not text:
        return ""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return text[:window_chars]
    start = max(0, idx - window_chars // 2)
    end = min(len(text), idx + len(keyword) + window_chars // 2)
    return text[start:end].strip()
