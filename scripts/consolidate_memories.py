
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from core import db
from core.hyperbolic import frechet_mean
from core.hyperbolic_clustering import cluster_hyperbolic

def consolidate():
    similarity_threshold = getattr(config, "MEMORY_CONSOLIDATION_THRESHOLD", 0.5)
    max_distance = (1.0 / similarity_threshold) - 1.0 if similarity_threshold > 0 else 0.0

    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("SELECT memory_id, content, embedding FROM memory_entries WHERE embedding_space='hyperbolic'")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        print("No hyperbolic memories to consolidate.")
        return

    embeddings = []
    contents = []
    ids = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        embeddings.append(emb)
        contents.append(row["content"])
        ids.append(row["memory_id"])

    clusters = cluster_hyperbolic(embeddings, max_dist=max_distance)
    merged_count = 0
    conn = db.db_connect("memories")
    cur = conn.cursor()
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        cluster_ids = [ids[i] for i in cluster]
        cluster_embs = [embeddings[i] for i in cluster]
        cluster_contents = [contents[i] for i in cluster]
        centroid = frechet_mean(cluster_embs, steps=10)
        snippet = " ; ".join(c[:200] for c in cluster_contents[:5])
        content = f"[Consolidated memory of {len(cluster_ids)} related memories]: {snippet}"
        blob = centroid.tobytes()
        cur.execute("""
            INSERT INTO memory_entries (session_id, memory_type, content, importance, embedding, embedding_space)
            VALUES ('consolidated', 'consolidated', ?, 1.0, ?, 'hyperbolic')
        """, (content, blob))
        for mid in cluster_ids:
            cur.execute("DELETE FROM memory_keywords WHERE memory_id=?", (mid,))
            cur.execute("DELETE FROM memory_entries WHERE memory_id=?", (mid,))
        merged_count += 1
    conn.commit()
    conn.close()
    print(f"Consolidated {merged_count} clusters of memories.")

if __name__ == "__main__":
    consolidate()
