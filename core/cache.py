import sqlite3

import config
from core import db


def init_ocr_cache():
    """Create ocr_cache table if it doesn't exist."""
    conn = db.db_connect("ocr")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ocr_cache (
            file_hash TEXT,
            pages INTEGER,
            dpi INTEGER,
            ocr_text TEXT,
            PRIMARY KEY (file_hash, pages, dpi)
        )
    """)
    conn.commit()
    conn.close()


def get_cached_ocr(file_hash: str, pages: int, dpi: int) -> str | None:
    """Retrieve OCR text from cache."""
    conn = db.db_connect("ocr")
    cur = conn.cursor()
    cur.execute(
        "SELECT ocr_text FROM ocr_cache WHERE file_hash=? AND pages=? AND dpi=?",
        (file_hash, pages, dpi),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def cache_ocr(file_hash: str, pages: int, dpi: int, text: str):
    """Store OCR text in cache."""
    conn = db.db_connect("ocr")
    conn.execute(
        "INSERT OR REPLACE INTO ocr_cache (file_hash, pages, dpi, ocr_text) VALUES (?, ?, ?, ?)",
        (file_hash, pages, dpi, text),
    )
    conn.commit()
    conn.close()