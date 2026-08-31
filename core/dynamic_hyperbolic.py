
import numpy as np
from core.hyperbolic import hyperbolic_distance

def dynamic_radius(query_emb, candidate_embs, k=10, scale=1.2):
    """
    Compute a dynamic hyperbolic radius based on the distance to the k-th nearest candidate.
    If fewer than k candidates, use max distance (or 1.0 if none).
    """
    if not candidate_embs:
        return 1.0
    distances = [hyperbolic_distance(query_emb, emb) for emb in candidate_embs]
    distances.sort()
    if len(distances) >= k:
        d_k = distances[k-1]
    else:
        d_k = distances[-1]
    return float(scale * d_k)

def local_density(emb, all_embs, k=10):
    """
    Return average hyperbolic distance to k nearest neighbors.
    Lower value indicates denser region.
    """
    if not all_embs:
        return float('inf')
    distances = [hyperbolic_distance(emb, other) for other in all_embs if other is not emb]
    distances.sort()
    k = min(k, len(distances))
    if k == 0:
        return float('inf')
    return float(np.mean(distances[:k]))
