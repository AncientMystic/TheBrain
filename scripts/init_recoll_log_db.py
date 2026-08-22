"""
Initialize recoll_log database tables.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db

def init_recoll_log_db():
    conn = db.db_connect("recoll_log")  # will be added to DB_FILES or fallback to index
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recoll_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            purpose TEXT,
            generated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recoll_query_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            doc_hash TEXT NOT NULL,
            file_path TEXT,
            retrieved_at TEXT,
            FOREIGN KEY(query_id) REFERENCES recoll_queries(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS recoll_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER,
            doc_hash TEXT,
            processed_successfully INTEGER,
            processed_at TEXT,
            UNIQUE(query_id, doc_hash)
        )
    """)
    conn.commit()
    conn.close()
    print("[init] recoll_log tables ready.")

if __name__ == "__main__":
    init_recoll_log_db()
