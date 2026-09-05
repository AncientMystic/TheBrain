"""
Sparse shape-constraint operators + family projections (Phase 2.1).

Implements monotone Dj (2 nnz/row), Edgeworth Hij (4 nnz/cell), trapezoidal
Tij (1 nnz/row) in CSR, with PAVA (isotonic), per-cell clip (57), interval
clip. No Dykstra loop yet (Phase 2.2). Used for confidence calibration
(not overwrite) in verification_manager. Generic, config-driven, preserves
hyperbolic retrieval (constraints act on confidence scalars, not embeddings).
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)


def build_monotone(m):
    """Forward-difference Dj: (m-1) x m, rows [-1, +1]. CSR dict."""
    import scipy.sparse as _sp
    if m < 2:
        return None
    rows = []
    cols = []
    data = []
    for t in range(m - 1):
        rows += [t, t]
        cols += [t, t + 1]
        data += [-1.0, 1.0]
    return _sp.csr_matrix((data, (rows, cols)), shape=(m - 1, m), dtype=np.float32)


def pav_regression(y):
    """Pool-adjacent-violators isotonic regression (non-decreasing). O(n)."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n == 0:
        return y.astype(np.float32)
    # Standard PAVA via block averaging
    blocks = [[float(v), 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        avg_i = blocks[i][0] / blocks[i][1]
        avg_j = blocks[i + 1][0] / blocks[i + 1][1]
        if avg_i <= avg_j:
            i += 1
        else:
            # Pool
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
    out = np.empty(n, dtype=np.float64)
    pos = 0
    for s, c in blocks:
        avg = s / c
        out[pos:pos + c] = avg
        pos += c
    return out.astype(np.float32)


def project_monotone_block(theta):
    """Orthogonal projection onto monotone cone Dj theta >= 0 via PAVA."""
    return pav_regression(theta)


def edgeworth_clip(theta_grid):
    """Per-cell clipping for supermodularity: theta[t+1,u+1]-theta[t+1,u]-theta[t,u+1]+theta[t,u] >= 0.

    theta_grid: 2D array (mi x mj). Applies closed-form half-space projection (57)
    per cell once (single sweep; Dykstra outer loop in Phase 2.2 iterates).
    Returns corrected grid (copy).
    """
    g = np.asarray(theta_grid, dtype=np.float64).copy()
    mi, mj = g.shape
    # a has [+1,-1,-1,+1] at corners, ||a||^2 = 4
    for t in range(mi - 1):
        for u in range(mj - 1):
            cross = g[t + 1, u + 1] - g[t + 1, u] - g[t, u + 1] + g[t, u]
            if cross < 0:
                adj = -cross / 4.0
                g[t + 1, u + 1] += adj
                g[t + 1, u] -= adj
                g[t, u + 1] -= adj
                g[t, u] += adj
    return g.astype(np.float32)


def trapezoid_clip(theta_vec, lower, upper):
    """Interval clipping per parameter: Lt <= theta <= Ut. 1 nnz/row."""
    t = np.asarray(theta_vec, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float32) if False else np.asarray(upper, dtype=np.float64)
    return np.minimum(np.maximum(t, lo), hi).astype(np.float32)


def calibrate_confidences(confidences, monotone_idx=None, trapezoid_bounds=None):
    """Calibrate confidence vector without overwriting order more than needed.

    - monotone_idx: indices that should be non-decreasing (e.g. sorted by support count).
      Applies PAVA to those positions only.
    - trapezoid_bounds: (lower, upper) arrays same length; clips.
    Returns calibrated copy in [0,1]. Generic, no doc-specific thresholds.
    """
    c = np.asarray(confidences, dtype=np.float32).copy()
    try:
        if trapezoid_bounds is not None:
            lo, hi = trapezoid_bounds
            c = trapezoid_clip(c, lo, hi)
        if monotone_idx is not None and len(monotone_idx) >= 2:
            sub = c[np.asarray(monotone_idx)]
            proj = project_monotone_block(sub)
            c[np.asarray(monotone_idx)] = proj
        c = np.clip(c, 0.0, 1.0)
    except Exception as e:
        logger.warning(f"Confidence calibration failed: {e}", exc_info=True)
    return c
