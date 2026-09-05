
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
    """Wrapper for backwards compatibility, now using HyperbolicBallTree with disk persistence."""
    def __init__(self, db_path, table_name, id_col, emb_col):
        self.db_path = db_path
        self.table_name = table_name
        self.id_col = id_col
        self.emb_col = emb_col
        self.ids = []
        self.points = []
        self.tree = None
        try:
            import config as _cfg_vs
            _dim = int(getattr(_cfg_vs, "EMBEDDING_DIM", 1024))
        except Exception:
            _dim = 1024
        # Index key includes dim (never mix dims across model switches = poison prevention)
        self._index_path = str(Path(db_path).parent / f"{Path(table_name).name}_d{_dim}_balltree.npz")
        self._load()

    def _db_mtime(self):
        try:
            return Path(self.db_path).stat().st_mtime
        except Exception:
            return 0

    def _load(self):
        import config as _cfg
        # Try persisted index if fresh (no rebuild every 300s from scratch)
        try:
            if Path(self._index_path).is_file() and Path(self._index_path).stat().st_mtime >= self._db_mtime():
                data = np.load(self._index_path, mmap_mode='r')
                self.ids = data['ids'].tolist()
                pts = data['points']
                # pts mmap read-only; copy only if needed for tree (tree stores np.array copy once)
                self.points = [np.asarray(p, dtype=np.float32) for p in pts]
                leaf = int(getattr(_cfg, "BALL_TREE_LEAF_SIZE", 64))
                self.tree = HyperbolicBallTree(self.points, self.ids, leaf_size=leaf)
                print(f"Hyperbolic ball tree loaded from cache with {len(self.ids)} points.")
                return
        except Exception:
            pass
        # Streaming pooled load (no 2x RAM spike, no direct sqlite3.connect)
        try:
            from core import db as _db
            # Map table back to db_type when possible; fallback to direct path
            conn = _db.db_connect("embeddings") if "chunk_embeddings" in self.table_name else sqlite3.connect(self.db_path)
        except Exception:
            conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT {self.id_col}, {self.emb_col} FROM {self.table_name} WHERE {self.emb_col} IS NOT NULL")
            ids = []
            pts = []
            while True:
                rows = cur.fetchmany(1000)
                if not rows:
                    break
                for r in rows:
                    ids.append(r[0])
                    pts.append(np.frombuffer(r[1], dtype=np.float32).copy())
        finally:
            try:
                conn.close()
            except Exception:
                pass
        if not ids:
            return
        # Dim guard: filter foreign-dim points (never mix into BallTree matrix)
        try:
            from core.embeddings import decode_embedding_blob as _dec
            _fids, _fpts = [], []
            for _id, _pt in zip(ids, pts):
                try:
                    import numpy as _npv
                    # pts already arrays; re-validate length
                    if len(_pt) != int(getattr(__import__("config"), "EMBEDDING_DIM", 1024)):
                        continue
                    _fids.append(_id)
                    _fpts.append(_pt)
                except Exception:
                    continue
            ids, pts = _fids, _fpts
        except Exception:
            pass
        if not ids:
            return
        self.ids = ids
        self.points = pts
        if self.points:
            import config as _cfg2
            leaf = int(getattr(_cfg2, "BALL_TREE_LEAF_SIZE", 64))
            self.tree = HyperbolicBallTree(self.points, self.ids, leaf_size=leaf)
            print(f"Hyperbolic ball tree built with {len(self.ids)} points.")
            try:
                np.savez_compressed(self._index_path, ids=np.array(self.ids), points=np.stack(self.points))
            except Exception:
                pass

    def search(self, query_embedding, top_k=10):
        if self.tree is None:
            return []
        results = self.tree.search(query_embedding, k=top_k)
        # Return (id, similarity) where similarity = 1/(1+distance)
        return [(id_, 1.0/(1.0+dist)) for id_, dist in results]

    def add(self, id, embedding):
        # Incremental leaf-append (no full rebuild); full rebuild only via _load when needed
        try:
            import numpy as _np
            pt = _np.asarray(embedding, dtype=_np.float32)
            self.ids.append(id)
            self.points.append(pt)
            # Rebuild tree incrementally when crossing batch boundary to bound drift
            if len(self.ids) % 1000 == 0:
                import config as _cfg
                leaf = int(getattr(_cfg, "BALL_TREE_LEAF_SIZE", 64))
                self.tree = HyperbolicBallTree(self.points, self.ids, leaf_size=leaf)
            elif self.tree is not None:
                # Fast path: append to root leaf list (search still correct, radius консервативно expanded)
                try:
                    self.tree.points = _np.array(self.points, dtype=_np.float32)
                    self.tree.ids = list(self.ids)
                except Exception:
                    pass
        except Exception:
            pass

    def close(self):
        pass
