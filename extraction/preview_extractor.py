import re
from pathlib import Path
import pymupdf as fitz  # PyMuPDF
from extractors.registry import extract_text_from_file
import config


def extract_preview(filepath: str, keyword: str, page_hint: int = None) -> str:
    """Extract a contextual preview around the keyword."""
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


def extract_pdf_preview(pdf_path: Path, keyword: str, page_hint: int = None) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return ""

    page_indices = range(doc.page_count)
    if page_hint is not None and 0 < page_hint <= doc.page_count:
        page_indices = [page_hint - 1]

    for idx in page_indices:
        text = doc[idx].get_text()
        if keyword.lower() in text.lower():
            return window_around_keyword(text, keyword, config.PREVIEW_CHAR_WINDOW)

    if doc.page_count > 0:
        return window_around_keyword(doc[0].get_text(), keyword, config.PREVIEW_CHAR_WINDOW)
    return ""


def window_around_keyword(text: str, keyword: str, window_chars: int) -> str:
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return text[:window_chars]
    start = max(0, idx - window_chars // 2)
    end = min(len(text), idx + len(keyword) + window_chars // 2)
    return text[start:end].strip()
