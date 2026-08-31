
"""
Hyperbolic geometry module for TheBrain.
Uses Poincaré ball model with exact Möbius operations.
All points are numpy arrays of shape (dim,).
"""
import numpy as np


def mobius_add(u, v):
    """Möbius addition in the Poincaré ball."""
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    u_norm2 = np.dot(u, u)
    v_norm2 = np.dot(v, v)
    inner = 2.0 * np.dot(u, v)
    denominator = 1.0 + inner + u_norm2 * v_norm2
    numerator = (1.0 + inner + v_norm2) * u + (1.0 - u_norm2) * v
    return numerator / denominator


def exp_map(v):
    """Exponential map from tangent space at origin to the ball."""
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    result = np.tanh(norm / 2.0) * v / norm
    return np.clip(result, -0.999999, 0.999999)


def log_map(x):
    """Logarithmic map at origin (inverse of exp_map)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.zeros_like(x)
    norm = min(norm, 0.999999)
    result = 2.0 * np.arctanh(norm) * x / norm
    return result


def hyperbolic_distance(u, v):
    """Hyperbolic distance in the Poincaré ball."""
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    diff = u - v
    num = 2.0 * np.dot(diff, diff)
    denom = (1.0 - np.dot(u, u)) * (1.0 - np.dot(v, v))
    arg = 1.0 + num / max(denom, 1e-12)
    arg = max(1.0, arg)
    return np.arccosh(arg)


def exp_mu(mu, v):
    """Exponential map at point mu for tangent vector v."""
    mu = np.asarray(mu, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return mu.copy()
    # Conformal factor at mu
    lambda_mu = 2.0 / (1.0 - np.dot(mu, mu) + 1e-8)
    coef = np.tanh(lambda_mu * norm_v / 2.0) / norm_v
    return mobius_add(mu, coef * v)


def log_mu(mu, x):
    """Logarithmic map at point mu."""
    mu = np.asarray(mu, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # (-mu) ⊕ x
    diff = mobius_add(-mu, x)
    norm_diff = np.linalg.norm(diff)
    if norm_diff == 0:
        return np.zeros_like(x)
    norm_diff = min(norm_diff, 0.999999)
    # Coefficient is (1 - ||mu||²) = 2 / lambda_mu
    coef = (1.0 - np.dot(mu, mu)) * np.arctanh(norm_diff) / norm_diff
    return coef * diff


def frechet_mean(vectors, steps=100, lr=0.1):
    """
    Compute Fréchet mean in the Poincaré ball using Riemannian gradient descent.
    vectors: list of numpy arrays (points in ball)
    """
    vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
    if not vectors:
        return None
    # Initialise at arithmetic mean in tangent space at origin
    tangent_mean = np.mean([log_map(v) for v in vectors], axis=0)
    mu = exp_map(tangent_mean)
    for _ in range(steps):
        # Gradient of squared distance is -2 log_mu(v)
        # So to minimise, move in positive direction of mean log_mu
        grad = np.mean([log_mu(mu, v) for v in vectors], axis=0)
        mu = exp_mu(mu, 2.0 * lr * grad)   # correct sign: move toward data
        # Clip to stay inside ball
        norm = np.linalg.norm(mu)
        if norm >= 1.0:
            mu = mu / norm * 0.999999
    return mu


def euclidean_to_hyperbolic(embeddings):
    """Convert Euclidean tangent vectors to hyperbolic points."""
    return [exp_map(v) for v in embeddings]


def hyperbolic_to_euclidean(points):
    """Convert hyperbolic points to Euclidean tangent vectors at origin."""
    return [log_map(p) for p in points]
