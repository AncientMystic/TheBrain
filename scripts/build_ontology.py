
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from core import db
from core.hyperbolic_clustering import cluster_hyperbolic

def main():
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("SELECT global_node_id, canonical_name, embedding FROM global_nodes WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        print("No hyperbolic node embeddings found.")
        return
    embeddings = [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    names = [r["canonical_name"] for r in rows]
    clusters = cluster_hyperbolic(embeddings, n_clusters=min(20, len(rows)))
    print(f"Discovered {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        members = [names[j] for j in cluster]
        print(f"  Cluster {i}: {', '.join(members[:5])} ...")

if __name__ == "__main__":
    main()
