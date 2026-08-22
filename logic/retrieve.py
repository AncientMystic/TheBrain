import numpy as np
from core import db
from core.embeddings import get_embedding


def retrieve_logic_modules(query, top_k=5):
    q_emb = get_embedding(query)
    if not q_emb:
        return []
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)

    conn = db.db_connect("logic")
    cur = conn.cursor()
    cur.execute("SELECT logic_id, name, category, summary, content, embedding FROM logic_modules")
    rows = cur.fetchall()
    conn.close()

    results = []
    for logic_id, name, category, summary, content, blob in rows:
        sim = 0.0
        if blob:
            emb = np.frombuffer(blob, dtype=np.float32)
            sim = float(np.dot(q, emb) / (q_norm * np.linalg.norm(emb) + 1e-8))
        results.append((sim, logic_id, name, category, summary, content))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]