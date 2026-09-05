"""
Family-block Dykstra loop for gated polyhedral cones (generic).

Projects onto intersection of half-spaces by iterating family projections
with correction vectors (Boyle-Dykstra). Each family projector is closed-form
(PAVA, cell clip, interval clip) from core.shape_constraints. No doc-specific
logic; tolerances via config.
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)


def dykstra_project(x0, projectors, max_iter=50, tol=1e-6):
    """Project x0 onto intersection via family projectors.

    projectors: list of callables p(x)->projected x (each idempotent onto its set).
    Returns projected copy. Generic Boyle-Dykstra with per-block corrections.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    # Correction vectors per block
    corrections = [np.zeros_like(x) for _ in projectors]
    for _ in range(max_iter):
        x_prev = x.copy()
        for i, proj in enumerate(projectors):
            try:
                y = proj(x + corrections[i])
                corrections[i] = x + corrections[i] - y
                x = y
            except Exception as e:
                logger.warning(f"Dykstra block {i} failed: {e}", exc_info=True)
                continue
        if float(np.linalg.norm(x - x_prev)) < tol:
            break
    return x.astype(np.float32)


def monotone_chain_projector():
    from core.shape_constraints import project_monotone_block

    def proj(x):
        return project_monotone_block(np.asarray(x, dtype=np.float32))
    return proj


def box_projector(lo=0.0, hi=1.0):
    def proj(x):
        return np.minimum(np.maximum(np.asarray(x, dtype=np.float64), lo), hi).astype(np.float32)
    return proj
