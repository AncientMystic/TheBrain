import sqlite3
from pathlib import Path

import config

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


def db_connect(db_type: str = "index") -> sqlite3.Connection:
    if db_type not in DB_FILES:
        raise ValueError(f"Unknown database type: {db_type}")
    conn = sqlite3.connect(DB_FILES[db_type], timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def table_exists(conn, table_name):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None
