"""Octonion proof (phase 67): determinism, norm, rotation, probe.

Synthetic only (no DB): identical inputs -> identical bytes; rotation
preserves norm; conjugation probe flags a negated pair and clears a
matching pair; blob round-trips at 8 float64.
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

import numpy as np


def test_octonion():
    from core import octonion as O
    rng = np.random.default_rng(3)
    dim = 32
    v1 = rng.normal(size=dim).astype(np.float32)
    v2 = rng.normal(size=dim).astype(np.float32) + 5.0
    from core.hyperbolic import exp_map
    e1 = np.asarray(exp_map(v1), dtype=np.float32)
    e2 = np.asarray(exp_map(v2), dtype=np.float32)

    a = O.signature(e1, popularity=100, entity_type="person", depth=0.9,
                    degree=5, confidence=0.9, n_sources=3, n_aliases=12)
    b = O.signature(e1, popularity=100, entity_type="person", depth=0.9,
                    degree=5, confidence=0.9, n_sources=3, n_aliases=12)
    assert a.tobytes() == b.tobytes(), "non-deterministic signature"
    assert len(a) == 8 and O.norm(a) > 0

    # geometry survives: same-entity pair closer than cross-entity pair
    a2 = O.signature(np.asarray(exp_map(
        (v1 + rng.normal(scale=0.01, size=dim)).astype(np.float32)),
        dtype=np.float32), popularity=100, entity_type="person", depth=0.9,
        degree=5, confidence=0.9, n_sources=3, n_aliases=12)
    c = O.signature(e2, popularity=1, entity_type="place", depth=0.2,
                    degree=0, confidence=0.5, n_sources=1, n_aliases=1)
    assert np.linalg.norm(a - a2) < np.linalg.norm(a - c), "geometry lost"

    # rotation preserves norm
    r = O.rotate(a, 3, 0.7)
    assert abs(O.norm(r) - O.norm(a)) < 1e-12, "rotation changed norm"

    # probe: matching pair clears, negated pair flags
    m_match, f_match = O.contradiction_probe(a, a2)
    assert f_match is False, (m_match, f_match)
    neg = O.conjugate(a)
    m_neg, f_neg = O.contradiction_probe(neg, a)
    assert f_neg is True and m_neg > 0, (m_neg, f_neg)

    # blob round-trip
    assert np.array_equal(O.from_blob(O.to_blob(a)), a)
    try:
        O.from_blob(b"short")
        raise AssertionError("bad dim accepted")
    except ValueError:
        pass
    print("test_octonion: OK (deterministic, norm, rotation, probe, blob)")


if __name__ == "__main__":
    test_octonion()
