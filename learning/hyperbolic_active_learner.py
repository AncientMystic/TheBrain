
import numpy as np
from core.hyperbolic import hyperbolic_distance

def select_uncertain_points(query_embedding, candidate_embeddings, n_select=5):
    """Select candidate points farthest from query in hyperbolic space (most uncertain)."""
    distances = [hyperbolic_distance(query_embedding, e) for e in candidate_embeddings]
    indices = np.argsort(distances)[-n_select:]
    return indices
