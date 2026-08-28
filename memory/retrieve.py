import math
import time
import numpy as np
from core import db
from core.embeddings import get_embedding
import config


def retrieve_memories(query, top_k=5, session_id=None):
    if getattr(config, "USE_HYPERBOLIC_MEMORY", True):
        from memory.hyperbolic_memory import retrieve_memories_hyperbolic
        return retrieve_memories_hyperbolic(query, top_k, session_id)

    q_emb = get_embedding(query)
    if not q_emb:
        return []
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)

    conn = db.db_connect("memories")
    cur = conn.cursor()
    if session_id:
        cur.execute("SELECT memory_id, content, memory_type, embedding, timestamp FROM memory_entries WHERE session_id=?",
                    (session_id,))
    else:
        cur.execute("SELECT memory_id, content, memory_type, embedding, timestamp FROM memory_entries")
    rows = cur.fetchall()
    conn.close()

    results = []
    for row in rows:
        memory_id = row["memory_id"]
        content = row["content"]
        memory_type = row["memory_type"]
        blob = row["embedding"]
        timestamp = row["timestamp"]

        sim = 0.0
        if blob:
            emb = np.frombuffer(blob, dtype=np.float32)
            sim = float(np.dot(q, emb) / (q_norm * np.linalg.norm(emb) + 1e-8))

        if config.MEMORY_DECAY_ENABLED:
            try:
                # timestamp is ISO string; parse to epoch seconds if possible
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp)
                age = time.time() - dt.timestamp()
            except Exception:
                age = 0.0
            sim *= math.exp(-config.MEMORY_DECAY_FACTOR * age)

        results.append((sim, memory_id, content, memory_type))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]
