
import numpy as np
import sqlite3
from pathlib import Path
import config
from core import db
from core.embeddings import get_embedding
from core.hyperbolic import exp_map, log_map, hyperbolic_distance, frechet_mean

def _batched_chunks(batch_size=512):
    """Yield batches of (chunk_id, doc_hash, chunk_text, tangent_vector)."""
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, doc_hash, chunk_text, embedding FROM chunk_embeddings")
    try:
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            batch = []
            for row in rows:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                # Assuming embedding is Euclidean; tangent vector = emb
                tangent = emb
                batch.append((row["chunk_id"], row["doc_hash"], row["chunk_text"], tangent))
            yield batch
    finally:
        conn.close()

def _init_centroids(n_clusters, batch_size=512):
    """Initialize centroids from first batch."""
    for batch in _batched_chunks(batch_size):
        if not batch:
            return None
        tangents = np.array([item[3] for item in batch], dtype=np.float32)
        rng = np.random.default_rng(42)
        indices = rng.choice(len(tangents), n_clusters, replace=False)
        return tangents[indices]
    return None

def build_streaming_topic_index(n_clusters=None, batch_size=512, max_epochs=5):
    """Streaming mini-batch k-means in tangent space."""
    if n_clusters is None:
        n_clusters = getattr(config, "TOPIC_INDEX_CLUSTERS", 20)

    print(f"  Building streaming topic index with {n_clusters} clusters...")
    centroids = _init_centroids(n_clusters, batch_size)
    if centroids is None:
        print("  No embeddings found.")
        return

    counts = np.zeros(n_clusters, dtype=np.float32)
    for epoch in range(max_epochs):
        print(f"    Epoch {epoch+1}/{max_epochs}")
        for batch in _batched_chunks(batch_size):
            tangents = np.array([item[3] for item in batch], dtype=np.float32)
            dists = np.linalg.norm(tangents[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroid = tangents[mask].mean(axis=0)
                    counts[k] += mask.sum()
                    centroids[k] = centroids[k] * 0.9 + new_centroid * 0.1

    hyperbolic_centroids = [exp_map(c) for c in centroids]

    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS stream_topic_centroids")
    cur.execute("DROP TABLE IF EXISTS stream_topic_members")
    cur.execute("""
        CREATE TABLE stream_topic_centroids (
            cluster_id INTEGER PRIMARY KEY,
            centroid BLOB,
            member_count INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE stream_topic_members (
            cluster_id INTEGER,
            chunk_id INTEGER,
            doc_hash TEXT,
            PRIMARY KEY (cluster_id, chunk_id)
        )
    """)
    for cid, hc in enumerate(hyperbolic_centroids):
        blob = sqlite3.Binary(hc.tobytes())
        cur.execute("INSERT INTO stream_topic_centroids (cluster_id, centroid, member_count) VALUES (?,?,0)", (cid, blob))

    # Assign memberships
    for batch in _batched_chunks(batch_size):
        tangents = np.array([item[3] for item in batch], dtype=np.float32)
        dists = np.linalg.norm(tangents[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for (chunk_id, doc_hash, _, _), lab in zip(batch, labels):
            cur.execute("INSERT INTO stream_topic_members (cluster_id, chunk_id, doc_hash) VALUES (?,?,?)", (int(lab), chunk_id, doc_hash))
            cur.execute("UPDATE stream_topic_centroids SET member_count = member_count + 1 WHERE cluster_id=?", (int(lab),))
    conn.commit()
    conn.close()
    print(f"  Built streaming topic index with {n_clusters} clusters.")

def load_stream_topic_index():
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("SELECT cluster_id, centroid FROM stream_topic_centroids")
    centroids = [(row["cluster_id"], np.frombuffer(row["centroid"], dtype=np.float32)) for row in cur.fetchall()]
    cur.execute("SELECT cluster_id, chunk_id, doc_hash FROM stream_topic_members")
    members = {}
    for row in cur.fetchall():
        members.setdefault(row["cluster_id"], []).append((row["chunk_id"], row["doc_hash"]))
    conn.close()
    return centroids, members

def query_stream_topic_index(query_emb_h, top_clusters=5, chunks_per_cluster=3):
    centroids, members = load_stream_topic_index()
    if not centroids:
        return []
    distances = []
    for cid, centroid_h in centroids:
        d = hyperbolic_distance(query_emb_h, centroid_h)
        distances.append((d, cid))
    distances.sort(key=lambda x: x[0])
    results = []
    for d, cid in distances[:top_clusters]:
        cluster_members = members.get(cid, [])
        for chunk_id, doc_hash in cluster_members[:chunks_per_cluster]:
            conn = db.db_connect("index")
            cur = conn.cursor()
            cur.execute("SELECT chunk_text FROM document_chunks WHERE chunk_id=?", (chunk_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                results.append({
                    "chunk_id": chunk_id,
                    "doc_hash": doc_hash,
                    "text": row["chunk_text"][:500],
                    "cluster_id": cid,
                })
    return results
