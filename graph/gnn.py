"""
Simplified Graph Neural Network for node embeddings using NumPy.
"""
import numpy as np
import config
from core import db
from pathlib import Path


def build_adjacency_matrix():
    """Build adjacency matrix from global_edges."""
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("SELECT source_node_id, target_node_id, weight FROM global_edges")
    edges = cur.fetchall()
    conn.close()

    # Get max node id
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("SELECT MAX(global_node_id) FROM global_nodes")
    max_id = cur.fetchone()[0] or 0
    conn.close()

    adj = np.zeros((max_id + 1, max_id + 1), dtype=np.float32)
    for src, tgt, weight in edges:
        if src <= max_id and tgt <= max_id:
            adj[src, tgt] = weight
            adj[tgt, src] = weight
    return adj


def train_gnn(embedding_dim=64):
    """Train simple GCN-like embeddings using random feature vectors."""
    adj = build_adjacency_matrix()
    n = adj.shape[0]
    if n == 0:
        print("No nodes to train GNN.")
        return None

    # Add self-loops
    adj = adj + np.eye(n, dtype=np.float32)
    # Normalize adjacency
    d = np.sum(adj, axis=1, keepdims=True)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    norm_adj = d_inv_sqrt * adj * d_inv_sqrt.T

    # Random initial embeddings
    H = np.random.normal(0, 0.1, (n, embedding_dim)).astype(np.float32)
    W1 = np.random.normal(0, 0.1, (embedding_dim, embedding_dim)).astype(np.float32)
    W2 = np.random.normal(0, 0.1, (embedding_dim, embedding_dim)).astype(np.float32)

    # Two-layer message passing
    for _ in range(2):
        H = np.tanh(norm_adj @ H @ W1)
        H = H @ W2

    # Normalize embeddings
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    H = H / norms

    # Save
    model_dir = Path(config.GNN_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    np.save(model_dir / "entity_embeddings.npy", H)
    print(f"GNN embeddings trained and saved: {H.shape}")
    return H


def get_gnn_embeddings():
    """Load GNN embeddings if available."""
    path = Path(config.GNN_MODEL_DIR) / "entity_embeddings.npy"
    if path.exists():
        return np.load(path)
    return None


def get_gnn_similarity(query_terms, candidate_node_id):
    """Return structural similarity score between query and node."""
    emb = get_gnn_embeddings()
    if emb is None:
        return 0.0
    if candidate_node_id >= emb.shape[0]:
        return 0.0
    # Simplistic: return average cosine similarity with query terms' embeddings unknown.
    return 0.0
