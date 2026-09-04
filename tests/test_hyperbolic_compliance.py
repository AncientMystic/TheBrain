"""Hyperbolic geometry compliance — guards against lazy Euclidean fallbacks and double-mapping."""
import numpy as np

from core.hyperbolic import (
    ensure_hyperbolic,
    exp_map,
    hyperbolic_distance,
    hyperbolic_similarity,
)


def _rand_vec(dim=16, scale=1.0):
    return (np.random.default_rng(42).standard_normal(dim).astype(np.float32) * scale)


def test_ensure_hyperbolic_idempotent():
    v = _rand_vec(scale=0.3)
    h = exp_map(v)
    assert float(np.linalg.norm(h)) < 1.0
    h2 = ensure_hyperbolic(h, space='hyperbolic')
    np.testing.assert_allclose(h, h2, rtol=1e-5, atol=1e-6)


def test_no_double_map_drift():
    v = _rand_vec(scale=0.5)
    single = exp_map(v)
    double = exp_map(single)  # what buggy callers did — must differ from single
    single_fixed = ensure_hyperbolic(single, space='hyperbolic')
    # fixed path must NOT drift like double map
    np.testing.assert_allclose(single, single_fixed, rtol=1e-5, atol=1e-6)
    # double-mapping distorts: must be measurably different (smaller, since tanh compresses twice)
    assert float(np.linalg.norm(double - single)) > 1e-4


def test_distance_self_zero_and_symmetry():
    a = ensure_hyperbolic(_rand_vec(scale=0.4), space='hyperbolic')
    b = ensure_hyperbolic(_rand_vec(scale=0.6), space='hyperbolic')
    assert abs(float(hyperbolic_distance(a, a))) < 1e-5
    assert abs(float(hyperbolic_distance(a, b)) - float(hyperbolic_distance(b, a))) < 1e-5


def test_similarity_range():
    a = ensure_hyperbolic(_rand_vec(scale=0.4), space='hyperbolic')
    b = ensure_hyperbolic(_rand_vec(scale=0.4), space='hyperbolic')
    s = hyperbolic_similarity(a, b)
    assert 0.0 < s <= 1.0
    assert abs(hyperbolic_similarity(a, a) - 1.0) < 1e-5
