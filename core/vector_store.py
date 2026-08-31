
"""
Scalable hyperbolic vector store using a recursive ball tree.
Provides logarithmic-time nearest neighbour search in the Poincaré ball.
"""

import numpy as np
import sqlite3
from pathlib import Path
import config
from core.hyperbolic import hyperbolic_distance, frechet_mean, log_map, exp_map

class HyperbolicBallTree:
    def __init__(self, points, ids, leaf_size=32):
        self.points = np.array(points, dtype=np.float32)
        self.ids = list(ids)
        self.leaf_size = leaf_size
        self.tree = self._build(np.arange(len(self.ids)))

    def _build(self, indices):
        if len(indices) <= self.leaf_size:
            centroid = self._compute_centroid(indices)
            radius = max(hyperbolic_distance(self.points[i], centroid) for i in indices) if len(indices) > 0 else 0.0
            return {'indices': indices, 'centroid': centroid, 'radius': radius, 'left': None, 'right': None}

        # Split in tangent space (origin) using coordinate with largest variance
        tangent = np.array([log_map(self.points[i]) for i in indices])
        variances = np.var(tangent, axis=0)
        split_dim = int(np.argmax(variances))
        median = np.median(tangent[:, split_dim])
        left_indices = [i for i in indices if log_map(self.points[i])[split_dim] <= median]
        right_indices = [i for i in indices if log_map(self.points[i])[split_dim] > median]
        if not left_indices or not right_indices:
            # Force split if too many points
            left_indices = indices[:len(indices)//2]
            right_indices = indices[len(indices)//2:]
        centroid = self._compute_centroid(indices)
        radius = max(hyperbolic_distance(self.points[i], centroid) for i in indices)
        left = self._build(left_indices)
        right = self._build(right_indices)
        return {'indices': indices, 'centroid': centroid, 'radius': radius, 'left': left, 'right': right}

    def _compute_centroid(self, indices):
        if len(indices) == 0:
            return np.zeros_like(self.points[0])
        if len(indices) == 1:
            return self.points[indices[0]].copy()
        # Use frechet_mean on subset (bounded size)
        subset = self.points[indices[:100]]
        return frechet_mean(subset, steps=10)

    def search(self, query, k=10):
        query = np.asarray(query, dtype=np.float32)
        best = []  # list of (distance, id)
        self._search(self.tree, query, k, best)
        best.sort(key=lambda x: x[0])
        return [(id_, dist) for dist, id_ in best[:k]]

    def _search(self, node, query, k, best):
        if node is None:
            return
        # Prune if possible
        if len(best) >= k:
            worst_dist = best[-1][0]
            centroid_dist = hyperbolic_distance(query, node['centroid'])
            if centroid_dist - node['radius'] > worst_dist:
                return
        if node['left'] is None and node['right'] is None:
            # Leaf
            for idx in node['indices']:
                dist = hyperbolic_distance(query, self.points[idx])
                if len(best) < k:
                    best.append((dist, self.ids[idx]))
                    best.sort(key=lambda x: x[0])
                elif dist < best[-1][0]:
                    best[-1] = (dist, self.ids[idx])
                    best.sort(key=lambda x: x[0])
        else:
            # Decide order
            left_dist = hyperbolic_distance(query, node['left']['centroid']) if node['left'] else float('inf')
            right_dist = hyperbolic_distance(query, node['right']['centroid']) if node['right'] else float('inf')
            if left_dist < right_dist:
                self._search(node['left'], query, k, best)
                self._search(node['right'], query, k, best)
            else:
                self._search(node['right'], query, k, best)
                self._search(node['left'], query, k, best)

class ExactVectorStore:
    """Wrapper for backwards compatibility, now using HyperbolicBallTree."""
    def __init__(self, db_path, table_name, id_col, emb_col):
        self.db_path = db_path
        self.table_name = table_name
        self.id_col = id_col
        self.emb_col = emb_col
        self.ids = []
        self.points = []
        self.tree = None
        self._load()

    def _load(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT {self.id_col}, {self.emb_col} FROM {self.table_name} WHERE {self.emb_col} IS NOT NULL")
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return
        self.ids = [row[0] for row in rows]
        self.points = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
        if self.points:
            self.tree = HyperbolicBallTree(self.points, self.ids, leaf_size=64)
            print(f"Hyperbolic ball tree built with {len(self.ids)} points.")

    def search(self, query_embedding, top_k=10):
        if self.tree is None:
            return []
        results = self.tree.search(query_embedding, k=top_k)
        # Return (id, similarity) where similarity = 1/(1+distance)
        return [(id_, 1.0/(1.0+dist)) for id_, dist in results]

    def add(self, id, embedding):
        # For simplicity, ignore incremental adds; rebuild from scratch after ingestion
        pass

    def close(self):
        pass
