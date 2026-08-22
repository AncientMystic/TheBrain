import sqlite3
import numpy as np
from core import db
from core.embeddings import get_embedding


def store_memory(session_id, content, memory_type="fact", importance=0.5):
    emb = get_embedding(content)
    blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes()) if emb else None
    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO memory_entries (session_id, memory_type, content, importance, embedding)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, memory_type, content, importance, blob))
    memory_id = cur.lastrowid
    words = set(content.lower().split())
    for w in words:
        if len(w) > 3:
            cur.execute("INSERT OR IGNORE INTO memory_keywords (memory_id, keyword, weight) VALUES (?, ?, 1.0)",
                        (memory_id, w))
    conn.commit()
    conn.close()
    return memory_id