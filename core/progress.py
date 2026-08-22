import sqlite3
from datetime import datetime, timezone

import config
from core import db


def init_progress_table():
    """Create processing_progress table in index.db if needed."""
    conn = db.db_connect("index")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processing_progress (
            file_hash TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'processed',
            stage TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


class ProgressTracker:
    """
    Tracks processed file hashes in SQLite.
    Replaces JSON checkpoint files entirely.
    """

    def __init__(self):
        init_progress_table()
        self.total_files = 0
        self.processed_count = 0

    def is_processed(self, file_hash: str) -> bool:
        """
        Return True only if the file has been successfully processed (status = 'processed').
        Files with status 'error' or any other status are considered not processed.
        """
        conn = db.db_connect("index")
        cur = conn.cursor()
        cur.execute(
            "SELECT status FROM processing_progress WHERE file_hash=?",
            (file_hash,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return False
        return row[0] == "processed"

    def mark_processed(self, file_hash: str, status: str = "processed", stage: str = None):
        """Insert or update a file's processing status."""
        conn = db.db_connect("index")
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT OR REPLACE INTO processing_progress (file_hash, status, stage, updated_at)
            VALUES (?, ?, ?, ?)
        """, (file_hash, status, stage, now))
        conn.commit()
        conn.close()

    def mark_error(self, file_hash: str, stage: str = None):
        """Mark a file as errored (will NOT be skipped in future runs)."""
        self.mark_processed(file_hash, status="error", stage=stage)

    def reset(self):
        """Clear all progress."""
        conn = db.db_connect("index")
        conn.execute("DELETE FROM processing_progress")
        conn.commit()
        conn.close()