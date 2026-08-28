#!/usr/bin/env python3
"""
Migration: add embedding_space columns and gate_training_data table.
Safe to run multiple times.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db

def add_column_if_missing(conn, table, column, definition):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        print(f"Added {column} to {table}")
    else:
        print(f"{table}.{column} already exists")

def migrate():
    # 1. Add embedding_space to embeddings tables
    conn = db.db_connect("embeddings")
    add_column_if_missing(conn, "document_embeddings", "embedding_space", "TEXT DEFAULT 'euclidean'")
    add_column_if_missing(conn, "chunk_embeddings", "embedding_space", "TEXT DEFAULT 'euclidean'")
    add_column_if_missing(conn, "embedding_cache", "embedding_space", "TEXT DEFAULT 'euclidean'")
    conn.commit()
    conn.close()

    # 2. Add embedding_space to external_graph.global_nodes
    conn = db.db_connect("external_graph")
    add_column_if_missing(conn, "global_nodes", "embedding_space", "TEXT DEFAULT 'euclidean'")
    conn.commit()
    conn.close()

    # 3. Create gate_training_data table in key_facts.db (idempotent)
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gate_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_hash TEXT,
            features BLOB,
            label INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()