"""Klein maps roundtrip + Dykstra intersection."""
import numpy as np
from core.klein import poincare_to_klein, klein_to_poincare
from core.dykstra import dykstra_project, monotone_chain_projector, box_projector


def test_klein_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        v = rng.standard_normal(8).astype(np.float64) * 0.2
        k = poincare_to_klein(v)
        back = klein_to_poincare(k)
        assert np.allclose(v, back, atol=1e-6)
        assert float(np.dot(k, k)) < 1.0


def test_dykstra_monotone_box():
    y = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
    out = dykstra_project(y, [monotone_chain_projector(), box_projector(0.0, 1.0)], max_iter=50)
    for i in range(len(out) - 1):
        assert out[i] <= out[i + 1] + 1e-5
    assert all(0.0 - 1e-6 <= v <= 1.0 + 1e-6 for v in out)
