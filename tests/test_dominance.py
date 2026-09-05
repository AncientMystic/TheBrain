"""Dominance projection: slopes of p dominate q."""
import numpy as np
from core.shape_constraints import project_dominance


def test_already_dominant_unchanged():
    p = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    q = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    po, qo = project_dominance(p, q)
    assert np.allclose(po, p, atol=1e-6)
    assert np.allclose(qo, q, atol=1e-6)


def test_violation_corrected():
    p = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    q = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    po, qo = project_dominance(p, q)
    for t in range(len(po) - 1):
        dp = float(po[t + 1] - po[t])
        dq = float(qo[t + 1] - qo[t])
        assert dp + 1e-5 >= dq
