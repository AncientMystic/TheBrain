
import numpy as np
from core import db
from core.embeddings import get_embedding
from core.hyperbolic import exp_map, log_map, frechet_mean, hyperbolic_distance

def _get_messages(session_id, max_turns=30):
    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content FROM (
            SELECT role, content, timestamp
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ) ORDER BY timestamp ASC
    """, (session_id, max_turns * 2))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def _embed_messages(messages):
    result = []
    for msg in messages:
        emb = get_embedding(msg["content"])
        if emb is not None:
            h_emb = exp_map(np.array(emb, dtype=np.float32))
            result.append((msg, h_emb))
    return result

def cluster_messages_hyperbolic(messages, n_clusters=3):
    embedded = _embed_messages(messages)
    if len(embedded) <= n_clusters:
        return [msg for msg, _ in embedded]

    X = np.array([log_map(h) for _, h in embedded], dtype=np.float32)
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(len(X), n_clusters, replace=False)]
    for _ in range(20):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in range(n_clusters)])
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    representatives = []
    for k in range(n_clusters):
        indices = [i for i, lab in enumerate(labels) if lab == k]
        if not indices:
            continue
        centroid_h = exp_map(centroids[k])
        best_idx = min(indices, key=lambda i: hyperbolic_distance(embedded[i][1], centroid_h))
        representatives.append(embedded[best_idx][0])
    return representatives

def get_hyperbolic_conversation_context(session_id, max_recent=5, max_clusters=3):
    messages = _get_messages(session_id)
    if not messages:
        return ""
    recent = messages[-max_recent:]
    older = messages[:-max_recent]

    parts = []
    if older:
        representatives = cluster_messages_hyperbolic(older, n_clusters=max_clusters)
        for rep in representatives:
            role = "User" if rep["role"] == "user" else "Assistant"
            text = rep["content"][:300]
            parts.append(f"[Earlier {role}] {text}")

    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}")

    return "

".join(parts)
