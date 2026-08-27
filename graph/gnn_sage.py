"""
GraphSAGE model for TheBrain.

Pure PyTorch implementation with sparse neighbor sampling.
Node features are derived from text embeddings of canonical names and aliases.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import config
from pathlib import Path


class SparseGraphSAGE(nn.Module):
    """Two-layer GraphSAGE with mean aggregation."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(input_dim * 2, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: node features (num_nodes, input_dim)
        edge_index: (2, num_edges) with undirected edges.
        """
        # First layer aggregation
        src, dst = edge_index
        # For each node, aggregate neighbor features (mean)
        num_nodes = x.size(0)
        # Use scatter_mean
        neighbor_sum = torch.zeros_like(x)
        neighbor_count = torch.zeros(num_nodes, 1, device=x.device)
        neighbor_sum = neighbor_sum.index_add(0, dst, x[src])
        neighbor_count = neighbor_count.index_add(0, dst, torch.ones_like(dst, dtype=torch.float32).unsqueeze(1))
        # Add reverse direction
        neighbor_sum = neighbor_sum.index_add(0, src, x[dst])
        neighbor_count = neighbor_count.index_add(0, src, torch.ones_like(src, dtype=torch.float32).unsqueeze(1))
        neighbor_mean = neighbor_sum / torch.clamp(neighbor_count, min=1)

        # Concatenate self + neighbor mean
        h = torch.cat([x, neighbor_mean], dim=1)
        h = F.relu(self.layer1(h))
        h = self.dropout(h)

        # Second layer
        neighbor_sum2 = torch.zeros_like(h)
        neighbor_count2 = torch.zeros(num_nodes, 1, device=x.device)
        neighbor_sum2 = neighbor_sum2.index_add(0, dst, h[src])
        neighbor_count2 = neighbor_count2.index_add(0, dst, torch.ones_like(dst, dtype=torch.float32).unsqueeze(1))
        neighbor_sum2 = neighbor_sum2.index_add(0, src, h[dst])
        neighbor_count2 = neighbor_count2.index_add(0, src, torch.ones_like(src, dtype=torch.float32).unsqueeze(1))
        neighbor_mean2 = neighbor_sum2 / torch.clamp(neighbor_count2, min=1)
        h2 = torch.cat([h, neighbor_mean2], dim=1)
        h2 = self.layer2(h2)
        return F.normalize(h2, p=2, dim=1)


def load_gnn_model(model_dir: Optional[str] = None) -> Optional[SparseGraphSAGE]:
    """Load a trained GNN model if exists."""
    if model_dir is None:
        model_dir = Path(config.GNN_MODEL_DIR)
    model_dir = Path(model_dir)
    model_path = model_dir / "gnn_sage.pt"
    if not model_path.exists():
        return None

    # Determine input dimension from saved state
    state = torch.load(model_path, map_location='cpu')
    input_dim = state['layer1.weight'].shape[1] // 2  # layer1 input is input_dim*2
    hidden_dim = state['layer1.weight'].shape[0]
    output_dim = state['layer2.weight'].shape[0]

    model = SparseGraphSAGE(input_dim, hidden_dim, output_dim)
    model.load_state_dict(state)
    model.eval()
    return model


def get_gnn_embeddings(model_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """Load precomputed node embeddings."""
    if model_dir is None:
        model_dir = Path(config.GNN_MODEL_DIR)
    emb_path = Path(model_dir) / "node_embeddings.npy"
    if emb_path.exists():
        return np.load(emb_path)
    return None


def compute_gnn_embeddings(model: SparseGraphSAGE, x: torch.Tensor, edge_index: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """Compute embeddings for all nodes in batches."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            end = min(i + batch_size, x.size(0))
            # For simplicity, we compute on full graph each time, but can optimize later.
            # Here we do full forward (graph is small for now)
            # To be scalable, we'd use neighbor sampling. For now, full forward.
            embeddings.append(model(x, edge_index)[i:end])
    return torch.cat(embeddings, dim=0)
