from core import db


def is_file_processed(file_hash: str) -> bool:
    """Check if file hash exists in processing_progress."""
    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM processing_progress WHERE file_hash=?", (file_hash,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def mark_file_processed(file_hash: str, status: str = "processed", stage: str = None):
    """Mark file as processed in SQLite."""
    from core.progress import ProgressTracker
    tracker = ProgressTracker()
    tracker.mark_processed(file_hash, status=status, stage=stage)