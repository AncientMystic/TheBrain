"""
Klein-Poincare maps for constraint projection (generic, no doc-specific logic).

Klein geodesics are straight chords (half-spaces stay convex); Poincare
geodesics are orthogonal arcs (same paths, different visualization).
Use Klein where constraints stay linear, convert back for retrieval.
"""
import numpy as np


def poincare_to_klein(x):
    x = np.asarray(x, dtype=np.float64)
    n2 = float(np.dot(x, x))
    if n2 >= 1.0:
        x = x / (np.sqrt(n2) + 1e-12) * 0.999999
        n2 = float(np.dot(x, x))
    return (2.0 * x) / (1.0 + n2)


def klein_to_poincare(y):
    y = np.asarray(y, dtype=np.float64)
    n2 = float(np.dot(y, y))
    if n2 >= 1.0:
        y = y / (np.sqrt(n2) + 1e-12) * 0.999999
        n2 = float(np.dot(y, y))
    return y / (1.0 + np.sqrt(max(1e-12, 1.0 - n2)))
