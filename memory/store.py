import sqlite3
import numpy as np
import config
from core import db
from core.embeddings import get_embedding, get_embeddings_dict
import logging
logger = logging.getLogger(__name__)


def store_memories_batched(items):
    """Bulk store [(session_id, content, memory_type, importance)] with single embed batch + single txn.

    Generic, no doc-specific logic. Falls back to hyperbolic path per item for space consistency.
    """
    if not items:
        return []
    if getattr(config, "USE_HYPERBOLIC_MEMORY", True):
        from memory.hyperbolic_memory import store_memory_hyperbolic
        # Hyperbolic path already single-embed per item; batch via dict to avoid N HTTP
        texts = [c for _, c, _, _ in items]
        emb_map = get_embeddings_dict([t for t in texts if t], space='hyperbolic')
        ids = []
        # Reuse single connection via hyperbolic helper is per-item; keep simple loop here
        # but embeddings already cached by get_embeddings_dict above (persistent + memory cache hit)
        for sid, content, mtype, imp in items:
            ids.append(store_memory_hyperbolic(sid, content, mtype, imp))
        return ids
    texts = [c for _, c, _, _ in items]
    emb_map = get_embeddings_dict([t for t in texts if t], space='hyperbolic')
    conn = db.db_connect("memories")
    cur = conn.cursor()
    ids = []
    kw_rows = []
    for sid, content, mtype, imp in items:
        emb = emb_map.get(content)
        blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes()) if emb is not None else None
        cur.execute("INSERT INTO memory_entries (session_id, memory_type, content, importance, embedding) VALUES (?, ?, ?, ?, ?)",
                    (sid, mtype, content, imp, blob))
        mid = cur.lastrowid
        ids.append(mid)
        for w in set(content.lower().split()):
            if len(w) > 3:
                kw_rows.append((mid, w))
    if kw_rows:
        cur.executemany("INSERT OR IGNORE INTO memory_keywords (memory_id, keyword, weight) VALUES (?, ?, 1.0)",
                        [(mid, w) for mid, w in kw_rows])
    conn.commit()
    conn.close()
    return ids


def store_memory(session_id, content, memory_type="fact", importance=0.5):
    if getattr(config, "USE_HYPERBOLIC_MEMORY", True):
        from memory.hyperbolic_memory import store_memory_hyperbolic
        return store_memory_hyperbolic(session_id, content, memory_type, importance)

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
