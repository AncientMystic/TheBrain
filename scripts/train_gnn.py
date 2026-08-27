#!/usr/bin/env python3
"""
Train GraphSAGE on external graph.

Usage:
    python scripts/train_gnn.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from core import db
from core.embeddings import get_embeddings_batch
from graph.gnn_sage import SparseGraphSAGE
import config


def load_graph():
    """Load nodes and edges from external_graph.db."""
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("SELECT global_node_id, canonical_name, node_type, aliases_json FROM global_nodes")
    nodes = cur.fetchall()
    cur.execute("SELECT source_node_id, target_node_id FROM global_edges")
    edges = cur.fetchall()
    conn.close()

    if not nodes:
        print("No nodes found.")
        return None, None, None, None

    # Prepare node texts
    node_texts = []
    node_ids = []
    for row in nodes:
        node_id, canonical, ntype, aliases_json = row
        aliases = []
        if aliases_json:
            import json
            try:
                aliases = json.loads(aliases_json)
            except:
                aliases = []
        text = canonical + " " + " ".join(aliases)
        node_texts.append(text)
        node_ids.append(node_id)

    print(f"Generating embeddings for {len(node_texts)} nodes...")
    node_emb = get_embeddings_batch(node_texts, batch_size=config.EMBEDDING_BATCH_SIZE)
    if any(e is None for e in node_emb):
        print("Warning: some node embeddings failed.")
        node_emb = [np.zeros(64) if e is None else e for e in node_emb]
    node_emb = np.array(node_emb, dtype=np.float32)

    # Map node_id to index
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = []
    for src, tgt in edges:
        if src in id_to_idx and tgt in id_to_idx:
            edge_index.append([id_to_idx[src], id_to_idx[tgt]])
    if not edge_index:
        print("No edges found.")
        return None, None, None, None
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return torch.tensor(node_emb), edge_index, node_ids, node_texts


def train():
    x, edge_index, node_ids, node_texts = load_graph()
    if x is None:
        return

    # Prepare data
    num_nodes = x.size(0)
    input_dim = x.size(1)
    model = SparseGraphSAGE(input_dim, hidden_dim=128, output_dim=64)
    optimizer = Adam(model.parameters(), lr=0.01)

    # Unsupervised link prediction: positive edges from graph, negative sampled
    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # Remove duplicate edges (optional)
    # Simple training loop
    model.train()
    epochs = 20
    batch_size = 256
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        # Sample positive edges
        pos_edges = edge_index_undirected
        pos_src, pos_dst = pos_edges
        # Generate negative samples
        neg_src = torch.randint(0, num_nodes, (pos_src.size(0),))
        neg_dst = torch.randint(0, num_nodes, (pos_dst.size(0),))

        # Compute embeddings for all nodes (could be expensive for large graphs; for now full)
        embeddings = model(x, edge_index_undirected)

        # Positive loss (dot product)
        pos_score = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)
        neg_score = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

        # Binary cross entropy with logits
        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_score, neg_score]), torch.cat([pos_labels, neg_labels]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}")

    # Save model and embeddings
    out_dir = Path(config.GNN_MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "gnn_sage.pt")
    print("Model saved to", out_dir / "gnn_sage.pt")

    # Compute final embeddings
    model.eval()
    with torch.no_grad():
        final_emb = model(x, edge_index_undirected)
    np.save(out_dir / "node_embeddings.npy", final_emb.numpy())
    print("Node embeddings saved to", out_dir / "node_embeddings.npy")
    # Also save node ids for mapping
    np.save(out_dir / "node_ids.npy", np.array(node_ids))
    print("Node IDs saved.")


if __name__ == "__main__":
    train()
