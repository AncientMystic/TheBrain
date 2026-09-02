
#!/usr/bin/env python3
"""
Train GraphSAGE on external graph using hyperbolic features.
Converts hyperbolic node embeddings to tangent space for input,
trains GNN in Euclidean tangent space, then converts GNN outputs back to hyperbolic.
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
from core.hyperbolic import log_map, exp_map
import config


def load_graph():
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

    node_texts = []
    node_ids = []
    for row in nodes:
        node_id, canonical, ntype, aliases_json = row
        aliases = []
        if aliases_json:
            import json
            try:
                aliases = json.loads(aliases_json)
            except Exception as e:
                aliases = []
        text = canonical + " " + " ".join(aliases)
        node_texts.append(text)
        node_ids.append(node_id)

    print(f"Generating hyperbolic embeddings for {len(node_texts)} nodes...")
    node_emb_hyper = get_embeddings_batch(node_texts, space='hyperbolic')
    node_emb_hyper = [np.array(e, dtype=np.float32) if e is not None else np.zeros(64) for e in node_emb_hyper]

    # Convert to tangent space at origin for Euclidean processing
    node_emb_tangent = np.array([log_map(e) for e in node_emb_hyper], dtype=np.float32)

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = []
    for src, tgt in edges:
        if src in id_to_idx and tgt in id_to_idx:
            edge_index.append([id_to_idx[src], id_to_idx[tgt]])
    if not edge_index:
        print("No edges found.")
        return None, None, None, None
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return torch.tensor(node_emb_tangent), edge_index, node_ids, node_emb_hyper


def train():
    x, edge_index, node_ids, node_emb_hyper = load_graph()
    if x is None:
        return

    num_nodes = x.size(0)
    input_dim = x.size(1)
    model = SparseGraphSAGE(input_dim, hidden_dim=128, output_dim=64)
    optimizer = Adam(model.parameters(), lr=0.01)

    edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    epochs = 20
    for epoch in range(epochs):
        model.train()
        pos_src, pos_dst = edge_index_undirected
        neg_src = torch.randint(0, num_nodes, (pos_src.size(0),))
        neg_dst = torch.randint(0, num_nodes, (pos_dst.size(0),))

        embeddings = model(x, edge_index_undirected)

        pos_score = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)
        neg_score = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

        pos_labels = torch.ones_like(pos_score)
        neg_labels = torch.zeros_like(neg_score)
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_score, neg_score]), torch.cat([pos_labels, neg_labels]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    out_dir = Path(config.GNN_MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "gnn_sage.pt")
    print("Model saved to", out_dir / "gnn_sage.pt")

    model.eval()
    with torch.no_grad():
        final_emb_tangent = model(x, edge_index_undirected)
    # Convert final GNN embeddings back to hyperbolic space
    final_emb_hyper = np.array([exp_map(e.numpy()) for e in final_emb_tangent], dtype=np.float32)
    np.save(out_dir / "node_embeddings.npy", final_emb_hyper)
    np.save(out_dir / "node_ids.npy", np.array(node_ids))
    print("Node embeddings saved in hyperbolic space.")


if __name__ == "__main__":
    train()
