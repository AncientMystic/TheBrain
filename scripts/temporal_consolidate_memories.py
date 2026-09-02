
import sys
from pathlib import Path
import numpy as np
import sqlite3
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from core import db
from core.hyperbolic import frechet_mean, hyperbolic_distance

def temporal_distance(emb1, t1, emb2, t2, time_weight=0.1):
    """Combined hyperbolic and temporal distance."""
    d_h = hyperbolic_distance(emb1, emb2)
    # t1, t2 are timestamps (strings); parse to seconds since epoch
    try:
        dt1 = datetime.fromisoformat(t1)
        dt2 = datetime.fromisoformat(t2)
        d_t = abs((dt1 - dt2).total_seconds())
    except Exception as e:
        d_t = 0
    return d_h + time_weight * d_t

def main():
    threshold = getattr(config, "MEMORY_CONSOLIDATION_THRESHOLD", 0.5)
    max_dist = (1.0 / threshold) - 1.0
    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("SELECT memory_id, content, embedding, timestamp FROM memory_entries WHERE embedding_space='hyperbolic'")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        print("No hyperbolic memories.")
        return
    # Greedy clustering with temporal distance
    clusters = []
    used = set()
    for i, row in enumerate(rows):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i+1, len(rows)):
            if j in used:
                continue
            d = temporal_distance(
                np.frombuffer(rows[i]["embedding"], dtype=np.float32), rows[i]["timestamp"],
                np.frombuffer(rows[j]["embedding"], dtype=np.float32), rows[j]["timestamp"]
            )
            if d < max_dist:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)
    merged_count = 0
    conn = db.db_connect("memories")
    cur = conn.cursor()
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        ids = [rows[i]["memory_id"] for i in cluster]
        embs = [np.frombuffer(rows[i]["embedding"], dtype=np.float32) for i in cluster]
        contents = [rows[i]["content"] for i in cluster]
        centroid = frechet_mean(embs, steps=10)
        snippet = " ; ".join(c[:200] for c in contents[:5])
        content = f"[Consolidated memory of {len(ids)} related memories]: {snippet}"
        blob = centroid.tobytes()
        cur.execute("""
            INSERT INTO memory_entries (session_id, memory_type, content, importance, embedding, embedding_space)
            VALUES ('consolidated', 'consolidated', ?, 1.0, ?, 'hyperbolic')
        """, (content, blob))
        for mid in ids:
            cur.execute("DELETE FROM memory_keywords WHERE memory_id=?", (mid,))
            cur.execute("DELETE FROM memory_entries WHERE memory_id=?", (mid,))
        merged_count += 1
    conn.commit()
    conn.close()
    print(f"Temporal consolidation merged {merged_count} clusters.")

if __name__ == "__main__":
    main()
