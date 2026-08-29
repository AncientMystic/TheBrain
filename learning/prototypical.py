
import numpy as np
from core.hyperbolic import frechet_mean, hyperbolic_distance

def compute_prototypes(embeddings_by_class):
    """Compute hyperbolic prototype for each class."""
    prototypes = {}
    for cls, embs in embeddings_by_class.items():
        prototypes[cls] = frechet_mean(embs, steps=10)
    return prototypes

def classify_hyperbolic(embedding, prototypes):
    """Return class with smallest hyperbolic distance to embedding."""
    best_cls = None
    best_dist = float('inf')
    for cls, proto in prototypes.items():
        d = hyperbolic_distance(embedding, proto)
        if d < best_dist:
            best_dist = d
            best_cls = cls
    return best_cls
