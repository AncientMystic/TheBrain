from datetime import datetime, timezone

import config
from core import db
from core.file_utils import get_file_hash


def store_document(conn, file_hash: str, file_path: str, filename: str,
                   file_format: str, text: str, metadata: dict,
                   ocr_used: bool = False, page_count: int = None):
    """
    Insert or replace document record in index.db.documents.
    """
    now = datetime.now(timezone.utc).isoformat()
    text_length = len(text)

    conn.execute("""
        INSERT OR REPLACE INTO documents
        (file_hash, file_path, filename, file_format, title, author, year,
         page_count, text_length, ocr_used, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file_hash,
        file_path,
        filename,
        file_format,
        metadata.get("title", ""),
        metadata.get("author", ""),
        metadata.get("year", ""),
        page_count,
        text_length,
        1 if ocr_used else 0,
        now,
    ))


def store_chunks(conn, doc_hash: str, chunks: list[str]):
    """
    Store document chunks in index.db.document_chunks.
    """
    rows = []
    for idx, chunk in enumerate(chunks):
        rows.append((doc_hash, idx, chunk, None, None))
    conn.executemany("""
        INSERT INTO document_chunks (doc_hash, chunk_index, chunk_text, start_offset, end_offset)
        VALUES (?, ?, ?, ?, ?)
    """, rows)


def get_chunks_by_doc(doc_hash: str) -> list[dict]:
    """Retrieve chunks for a document."""
    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_id, chunk_index, chunk_text, start_offset, end_offset
        FROM document_chunks
        WHERE doc_hash = ?
        ORDER BY chunk_index
    """, (doc_hash,))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]