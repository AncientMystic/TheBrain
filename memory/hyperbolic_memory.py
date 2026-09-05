
import sqlite3
import numpy as np
from pathlib import Path
import config
from core import db
from core.embeddings import get_embedding
from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance, frechet_mean, exp_mu, log_mu

def _memory_embeddings(session_id=None):
    """Return list of (memory_id, content, hyperbolic_embedding) for a session or all.
       Only retrieves memories stored with embedding_space='hyperbolic'."""
    conn = db.db_connect("memories")
    cur = conn.cursor()
    if session_id:
        cur.execute("SELECT memory_id, content, embedding FROM memory_entries WHERE session_id=? AND embedding_space='hyperbolic'", (session_id,))
    else:
        cur.execute("SELECT memory_id, content, embedding FROM memory_entries WHERE embedding_space='hyperbolic'")
    rows = cur.fetchall()
    conn.close()
    memories = []
    for row in rows:
        if row["embedding"] is not None:
            try:
                from core.embeddings import decode_embedding_blob as _dec2
                emb = _dec2(row["embedding"], context="hyperbolic_memory")
                if emb is None:
                    continue
            except Exception:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
            memories.append((row["memory_id"], row["content"], emb))
    return memories

def store_memory_hyperbolic(session_id, content, memory_type="fact", importance=0.5):
    emb = get_embedding(content)
    if emb is None:
        return None
    h_emb = ensure_hyperbolic(emb, space='hyperbolic')
    blob = sqlite3.Binary(h_emb.tobytes())
    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO memory_entries (session_id, memory_type, content, importance, embedding, embedding_space)
        VALUES (?, ?, ?, ?, ?, 'hyperbolic')
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

def retrieve_memories_hyperbolic(query, top_k=5, session_id=None):
    q_emb = get_embedding(query)
    if q_emb is None:
        return []
    q_h = ensure_hyperbolic(q_emb, space='hyperbolic')
    memories = _memory_embeddings(session_id)
    results = []
    for memory_id, content, emb in memories:
        d = hyperbolic_distance(q_h, emb)
        sim = 1.0 / (1.0 + d)
        results.append((sim, memory_id, content, "memory"))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def update_session_centroid(session_id, query_text, answer_text=""):
    """Update session centroid using weighted geodesic interpolation with decay."""
    q_emb = get_embedding(query_text)
    if q_emb is None:
        return
    q_h = ensure_hyperbolic(q_emb, space='hyperbolic')
    if answer_text:
        a_emb = get_embedding(answer_text)
        if a_emb is not None:
            a_h = ensure_hyperbolic(a_emb, space='hyperbolic')
            new_centroid = frechet_mean([q_h, a_h], steps=10)
        else:
            new_centroid = q_h
    else:
        new_centroid = q_h

    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("SELECT topic_centroid FROM memory_sessions WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    if row and row["topic_centroid"] is not None:
        old_centroid = np.frombuffer(row["topic_centroid"], dtype=np.float32)
        decay = getattr(config, "HYPERBOLIC_TOPIC_CENTROID_DECAY", 0.7)
        # Weighted geodesic interpolation: move (1-decay) toward new_centroid
        t = 1.0 - decay
        # combined = exp_old_centroid(t * log_old_centroid(new_centroid))
        combined = exp_mu(old_centroid, t * log_mu(old_centroid, new_centroid))
    else:
        combined = new_centroid
    blob = sqlite3.Binary(combined.tobytes())
    cur.execute("""
        INSERT INTO memory_sessions (session_id, topic_centroid)
        VALUES (?, ?)
        ON CONFLICT(session_id) DO UPDATE SET topic_centroid = excluded.topic_centroid
    """, (session_id, blob))
    conn.commit()
    conn.close()

def geodesic_memory_expansion(query_text, centroid_h, top_k=5):
    """Retrieve memories near the midpoint of query and centroid."""
    q_emb = get_embedding(query_text)
    if q_emb is None:
        return []
    q_h = ensure_hyperbolic(q_emb, space='hyperbolic')
    # Midpoint on geodesic: exp_q(0.5 * log_q(centroid))
    mid_h = exp_mu(q_h, 0.5 * log_mu(q_h, centroid_h))
    memories = _memory_embeddings()
    results = []
    for memory_id, content, emb in memories:
        d = hyperbolic_distance(mid_h, emb)
        sim = 1.0 / (1.0 + d)
        results.append((sim, memory_id, content, "memory"))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

def cluster_history_hyperbolic(session_id, max_clusters=3):
    """Placeholder for hyperbolic clustering of conversation history."""
    return []
