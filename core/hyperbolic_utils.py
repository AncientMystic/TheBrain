
import numpy as np
from core.hyperbolic import exp_map, log_map, hyperbolic_distance, mobius_add, exp_mu, log_mu
import logging
logger = logging.getLogger(__name__)

def safe_exp_map(emb):
    """Map Euclidean vector to hyperbolic, returning zero vector on error."""
    try:
        return exp_map(np.asarray(emb, dtype=np.float32))
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return np.zeros_like(np.asarray(emb, dtype=np.float32))

def hyperbolic_similarity(emb_a, emb_b):
    """Convert hyperbolic distance to similarity in [0,1]."""
    d = hyperbolic_distance(emb_a, emb_b)
    return float(1.0 / (1.0 + d))

def geodesic_interpolate(a, b, t=0.5):
    """Return point on geodesic between hyperbolic points a and b at parameter t."""
    return exp_mu(a, t * log_mu(a, b))
