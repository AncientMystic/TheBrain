"""Shape-constraint operators: monotone PAVA, Edgeworth clip, trapezoidal clip."""
from core.shape_constraints import (
    build_monotone,
    pav_regression,
    edgeworth_clip,
    trapezoid_clip,
    calibrate_confidences,
)
import numpy as np


def test_monotone_projection():
    y = np.array([0.9, 0.2, 0.8, 0.1], dtype=np.float32)
    p = pav_regression(y)
    for i in range(len(p) - 1):
        assert p[i] <= p[i + 1] + 1e-5


def test_edgeworth_clip():
    g = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    # cross = 0-0-0+1 = 1 >=0 already ok
    out = edgeworth_clip(g)
    assert out.shape == (2, 2)
    # violating cell: all zeros except top-left 0? make cross negative
    h = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    out2 = edgeworth_clip(h)
    cross = out2[1, 1] - out2[1, 0] - out2[0, 1] + out2[0, 0]
    assert cross >= -1e-5


def test_trapezoid_clip():
    v = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
    out = trapezoid_clip(v, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert list(out) == [0.0, 0.5, 1.0]


def test_calibrate_monotone():
    confs = [0.9, 0.2, 0.8]
    cal = calibrate_confidences(confs, monotone_idx=[0, 1, 2], trapezoid_bounds=([0, 0, 0], [1, 1, 1]))
    assert len(cal) == 3
    assert cal[0] <= cal[1] + 1e-5 and cal[1] <= cal[2] + 1e-5


def test_build_monotone_shape():
    m = build_monotone(5)
    assert m.shape == (4, 5)
    assert m.nnz == 8
