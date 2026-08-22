import numpy as np
from core import db
from core.embeddings import get_embedding


def retrieve_memories(query, top_k=5, session_id=None):
    q_emb = get_embedding(query)
    if not q_emb:
        return []
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)

    conn = db.db_connect("memories")
    cur = conn.cursor()
    if session_id:
        cur.execute("SELECT memory_id, content, memory_type, embedding FROM memory_entries WHERE session_id=?",
                    (session_id,))
    else:
        cur.execute("SELECT memory_id, content, memory_type, embedding FROM memory_entries")
    rows = cur.fetchall()
    conn.close()

    results = []
    for memory_id, content, memory_type, blob in rows:
        if blob:
            emb = np.frombuffer(blob, dtype=np.float32)
            sim = float(np.dot(q, emb) / (q_norm * np.linalg.norm(emb) + 1e-8))
            results.append((sim, memory_id, content, memory_type))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]