
import numpy as np
import sqlite3
from pathlib import Path
import config
from core import db
from core.embeddings import get_embeddings_batch
from core.hyperbolic import exp_map, frechet_mean, hyperbolic_distance
from core.hyperbolic_clustering import cluster_hyperbolic
import logging
logger = logging.getLogger(__name__)

def _load_chunks_for_index():
    """Load all chunk embeddings and texts from database."""
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, doc_hash, chunk_text, embedding FROM chunk_embeddings")
    rows = cur.fetchall()
    conn.close()
    return rows

def build_topic_index(max_clusters=None):
    """Cluster chunk embeddings in hyperbolic space and store centroids/members."""
    if max_clusters is None:
        max_clusters = getattr(config, "TOPIC_INDEX_CLUSTERS", 20)
    rows = _load_chunks_for_index()
    if not rows:
        print("  (No chunk embeddings to build topic index)")
        return

    # Convert embeddings to hyperbolic
    hyperbolic_embs = []
    valid_rows = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        try:
            h_emb = exp_map(emb)
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            continue
        hyperbolic_embs.append(h_emb)
        valid_rows.append(row)

    if not hyperbolic_embs:
        print("  (No valid hyperbolic embeddings)")
        return

    # Cluster
    clusters = cluster_hyperbolic(hyperbolic_embs, n_clusters=max_clusters)

    # Clear old index
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("DELETE FROM topic_index_centroids")
    cur.execute("DELETE FROM topic_index_members")
    conn.commit()
    conn.close()

    # Store new index
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    for cluster_id, cluster in enumerate(clusters):
        if not cluster:
            continue
        cluster_embs = [hyperbolic_embs[i] for i in cluster]
        centroid = frechet_mean(cluster_embs, steps=10)
        blob = sqlite3.Binary(centroid.tobytes())
        cur.execute(
            "INSERT INTO topic_index_centroids (cluster_id, centroid, member_count) VALUES (?,?,?)",
            (cluster_id, blob, len(cluster))
        )
        for idx in cluster:
            row = valid_rows[idx]
            cur.execute(
                "INSERT INTO topic_index_members (cluster_id, chunk_id, doc_hash) VALUES (?,?,?)",
                (cluster_id, row["chunk_id"], row["doc_hash"])
            )
    conn.commit()
    conn.close()
    print(f"  (Built topic index with {len(clusters)} clusters)")

def load_topic_index():
    """Return list of (cluster_id, centroid_emb) and dict cluster_id->list of member chunks."""
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("SELECT cluster_id, centroid FROM topic_index_centroids")
    centroids = []
    for row in cur.fetchall():
        centroid = np.frombuffer(row["centroid"], dtype=np.float32)
        centroids.append((row["cluster_id"], centroid))
    cur.execute("SELECT cluster_id, chunk_id, doc_hash FROM topic_index_members")
    members = {}
    for row in cur.fetchall():
        members.setdefault(row["cluster_id"], []).append((row["chunk_id"], row["doc_hash"]))
    conn.close()
    return centroids, members

def query_topic_index(query_emb, top_clusters=None, chunks_per_cluster=None):
    """Find nearest clusters to query and return representative chunks."""
    if top_clusters is None:
        top_clusters = getattr(config, "TOPIC_INDEX_CLUSTERS", 5)
    if chunks_per_cluster is None:
        chunks_per_cluster = getattr(config, "TOPIC_INDEX_CHUNKS_PER_CLUSTER", 3)

    centroids, members = load_topic_index()
    if not centroids:
        return []

    # Compute distances to centroids
    distances = []
    for cluster_id, centroid in centroids:
        d = hyperbolic_distance(query_emb, centroid)
        distances.append((d, cluster_id))
    distances.sort(key=lambda x: x[0])
    selected_clusters = distances[:top_clusters]

    results = []
    for _, cluster_id in selected_clusters:
        cluster_members = members.get(cluster_id, [])
        if not cluster_members:
            continue
        # For simplicity, take first chunks_per_cluster members (or sort by distance to query later)
        for chunk_id, doc_hash in cluster_members[:chunks_per_cluster]:
            # Fetch chunk text from index database
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
                    "cluster_id": cluster_id,
                })
    return results
