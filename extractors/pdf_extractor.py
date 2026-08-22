import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
import pytesseract
from PIL import Image

import config
from core import cache
from core.file_utils import get_file_hash
from core.text_utils import normalise_text


def _ocr_page(pix):
    try:
        img = Image.open(io.BytesIO(pix))
        return pytesseract.image_to_string(img, config='--psm 6')
    except Exception as e:
        print(f"      (OCR page error: {e})")
        return ""


def ocr_pdf_pages(pdf_path, max_pages=None, dpi=None, title_pages=None, title_dpi=None):
    """
    OCR all/some pages of a PDF. If max_pages is None, OCR all pages.
    Uses parallel rendering/OCR.
    """
    if dpi is None:
        dpi = config.OCR_DPI
    if title_dpi is None:
        title_dpi = config.TITLE_PAGE_DPI
    if title_pages is None:
        title_pages = min(config.TITLE_PAGE_COUNT, max_pages if max_pages else 3)

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)
    title_pages = min(title_pages, pages_to_process)

    print(f"    (Rendering {pages_to_process} pages, first {title_pages} at high DPI)...", end="", flush=True)
    images = []
    for page_num in range(pages_to_process):
        page = doc[page_num]
        if page_num < title_pages:
            pix = page.get_pixmap(dpi=title_dpi)
        else:
            pix = page.get_pixmap(dpi=dpi)
        images.append(pix.tobytes("png"))
    doc.close()
    print(" rendered.", end="", flush=True)

    print(" OCR'ing in parallel...", end="", flush=True)
    full_text = [""] * len(images)
    max_workers = min(os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_ocr_page, img_bytes): idx
            for idx, img_bytes in enumerate(images)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            full_text[idx] = future.result()
    print(" done.", flush=True)

    combined = "\n".join(full_text)
    combined = normalise_text(combined)
    # No truncation to MAX_TEXT_CHARS because we're indexing entire document
    return combined


def extract_pdf(filepath: Path) -> dict:
    """
    Extract text from PDF. Try PyMuPDF text extraction first.
    If extracted text is too short, OCR the entire PDF.
    Returns dict with text, metadata, format.
    """
    file_path = str(filepath)
    file_hash = get_file_hash(file_path)

    # Try direct text extraction (full document)
    text = ""
    ocr_used = False
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        text = normalise_text(text)
    except Exception as e:
        print(f"    (PyMuPDF error: {e})")
        text = ""

    # If text is insufficient, use OCR full document
    if len(text.strip()) < config.MIN_TEXT_CHARS_FOR_OCR_SKIP:
        print(f"    (Text extraction returned {len(text)} chars; running OCR on full document)")
        # Use cache if available
        cached = cache.get_cached_ocr(file_hash, pages=0, dpi=config.OCR_DPI)  # pages=0 means full
        if cached:
            text = cached
        else:
            text = ocr_pdf_pages(file_path, max_pages=None, dpi=config.OCR_DPI)
            cache.cache_ocr(file_hash, pages=0, dpi=config.OCR_DPI, text=text)
        ocr_used = True

    metadata = {
        "title": filepath.stem,
        "author": "",
        "year": "",
    }

    # Try to extract PDF metadata
    try:
        doc = fitz.open(file_path)
        pdf_meta = doc.metadata
        if pdf_meta:
            metadata["title"] = pdf_meta.get("title") or filepath.stem
            metadata["author"] = pdf_meta.get("author") or ""
            if pdf_meta.get("creationDate"):
                # Extract year if possible
                import re
                match = re.search(r'(19|20)\d{2}', pdf_meta.get("creationDate", ""))
                if match:
                    metadata["year"] = match.group(0)
        doc.close()
    except Exception:
        pass

    return {
        "text": text,
        "metadata": metadata,
        "format": "pdf",
        "ocr_used": ocr_used,
    }