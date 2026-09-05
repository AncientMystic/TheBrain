"""Convex unimodal surrogate + exact peak projection."""
import numpy as np
from core.shape_constraints import unimodal_penalty, project_unimodal


def test_penalty_zero_for_linear():
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(unimodal_penalty(y)) < 1e-6


def test_penalty_positive_for_wiggle():
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    assert unimodal_penalty(y) > 0.0


def test_projection_single_peak():
    y = np.array([0.0, 0.2, 1.0, 0.8, 0.9, 0.1], dtype=np.float32)
    p = project_unimodal(y)
    assert len(p) == len(y)
    k = int(np.argmax(p))
    # Non-decreasing up to peak, non-increasing after (tolerance)
    for i in range(k):
        assert p[i] <= p[i + 1] + 1e-5
    for i in range(k, len(p) - 1):
        assert p[i] >= p[i + 1] - 1e-5
