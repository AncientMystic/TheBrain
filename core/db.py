import sqlite3
import threading
import time
from functools import wraps
from pathlib import Path

import config

def with_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        delay = 0.5
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
                    continue
                raise
    return wrapper


_conn_local = threading.local()

class PooledConnection:
    """Wraps sqlite3.Connection so close() returns it to the pool."""
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        if not hasattr(_conn_local, "pool"):
            _conn_local.pool = {}
        db_type = getattr(self, "_db_type", None)
        if db_type:
            # Rollback any uncommitted transaction before returning to pool.
            try:
                self._conn.rollback()
            except Exception:
                pass
            _conn_local.pool[db_type] = self._conn

    def real_close(self):
        try:
            self._conn.close()
        except Exception:
            pass

DB_FILES = {
    "index": config.INDEX_DB_FILE,
    "summaries": config.SUMMARIES_DB_FILE,
    "key_facts": config.KEY_FACTS_DB_FILE,
    "embeddings": config.EMBEDDINGS_DB_FILE,
    "hypergraph": config.HYPERGRAPH_DB_FILE,
    "external_graph": config.EXTERNAL_GRAPH_DB_FILE,
    "ocr": config.OCR_CACHE_DB_FILE,
    "memories": config.MEMORIES_DB_FILE,
    "logic": config.LOGIC_DB_FILE,
    "reasoning": config.REASONING_DB_FILE,
    "recoll_log": config.RECOLL_LOG_DB_FILE,
    "verification_standards": config.VERIFICATION_STANDARDS_DB_FILE,
}

for _db_path in DB_FILES.values():
    Path(_db_path).parent.mkdir(parents=True, exist_ok=True)


def _make_connection(db_type: str) -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILES[db_type], timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")
    return conn


@with_retry
def db_connect(db_type: str = "index") -> sqlite3.Connection:
    if db_type not in DB_FILES:
        raise ValueError(f"Unknown database type: {db_type}")
    if not hasattr(_conn_local, "pool"):
        _conn_local.pool = {}
    if db_type in _conn_local.pool:
        conn = _conn_local.pool.pop(db_type)
    else:
        conn = _make_connection(db_type)
    pooled = PooledConnection(conn)
    pooled._db_type = db_type
    return pooled


def close_all_connections():
    """Close all pooled connections."""
    if hasattr(_conn_local, "pool"):
        for conn in _conn_local.pool.values():
            try:
                conn.close()
            except Exception:
                pass
        _conn_local.pool.clear()


def table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None
