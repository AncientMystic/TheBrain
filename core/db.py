import atexit
import queue
import sqlite3
import threading
import time
from functools import wraps
from pathlib import Path

import config
import logging
MAX_POOL_SIZE = 20  # maximum connections per database type
logger = logging.getLogger(__name__)

_pools: dict = {}
_pools_lock = threading.Lock()

def with_retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        delay = 0.5
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if (getattr(e, 'sqlite_errorcode', None) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                    and attempt < max_retries - 1):
                    time.sleep(delay * (2 ** attempt))
                    continue
                raise
    return wrapper


def _get_pool(db_type: str) -> "queue.Queue":
    with _pools_lock:
        q = _pools.get(db_type)
        if q is None:
            q = queue.Queue(maxsize=MAX_POOL_SIZE)
            _pools[db_type] = q
        return q


class PooledConnection:
    """Wraps sqlite3.Connection so close() returns it to the shared pool."""
    def __init__(self, conn, db_type: str):
        self._conn = conn
        self._db_type = db_type
        self._returned = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        if self._returned:
            return
        self._returned = True
        try:
            # Only rollback if caller left an open transaction (programming error).
            # Do not mask it as success: log loudly for audit.
            if getattr(self._conn, "in_transaction", False):
                logger.error("PooledConnection.close() with open transaction; rolling back to keep pool clean", exc_info=True)
                try:
                    self._conn.rollback()
                except Exception:
                    logger.warning("Rollback on pooled close failed", exc_info=True)
        finally:
            q = _get_pool(self._db_type)
            try:
                q.put_nowait(self._conn)
            except queue.Full:
                try:
                    self._conn.close()
                except Exception:
                    logger.warning("Overflow pool connection close failed", exc_info=True)

    def real_close(self):
        self._returned = True
        try:
            self._conn.close()
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            try:
                self._conn.commit()
            except Exception:
                logger.warning("Commit on context exit failed", exc_info=True)
        self.close()
        return False

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
    # check_same_thread=False so shared pool can hand connections across threads.
    # Callers must still use one connection at a time and return via close().
    conn = sqlite3.connect(DB_FILES[db_type], timeout=60, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    return conn


@with_retry
def db_connect(db_type: str = "index") -> sqlite3.Connection:
    """Return a pooled SQLite connection for the requested database type."""

    if db_type not in DB_FILES:
        raise ValueError(f"Unknown database type: {db_type}")
    q = _get_pool(db_type)
    try:
        conn = q.get_nowait()
    except queue.Empty:
        conn = _make_connection(db_type)
    # Validate connection is usable; recreate if closed/corrupt
    try:
        conn.execute("SELECT 1")
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        conn = _make_connection(db_type)
    return PooledConnection(conn, db_type)


def close_all_connections():
    """Close all pooled connections across all threads."""
    with _pools_lock:
        pools = list(_pools.items())
    for db_type, q in pools:
        while True:
            try:
                conn = q.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:
                logger.warning("Unexpected exception occurred", exc_info=True)
                pass


atexit.register(close_all_connections)


def table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None
