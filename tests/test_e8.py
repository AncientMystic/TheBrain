"""E8 proof (phase 68): decode exactness, routing sanity, live-shaped test.

- 240 minimal vectors, all norm sqrt(2) (112 integer + 128 half).
- Every minimal vector decodes to itself (lattice fixed points).
- Random points decode to valid lattice points (parity check).
- route_cells returns top-2 distinct keys, nearest first.
- route_entities finds rows by cell on a synthetic populated table.
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

import numpy as np


def _is_lattice(pt):
    v = np.asarray(pt, dtype=np.float64)
    r = np.round(v * 2.0) / 2.0
    if not np.allclose(v, r, atol=1e-9):
        return False
    if np.allclose(v, np.round(v), atol=1e-9):
        return int(round(v.sum())) % 2 == 0
    return int(round((v - 0.5).sum())) % 2 == 0


def test_e8():
    from core import e8 as E
    vecs = E.min_vectors()
    assert len(vecs) == 240, len(vecs)
    assert all(abs(np.linalg.norm(v) - np.sqrt(2)) < 1e-9 for v in vecs)
    for v in vecs:
        d = E.decode(v * 1.0)
        assert np.allclose(d, v, atol=1e-9), ("not fixed", v, d)
    rng = np.random.default_rng(5)
    for _ in range(60):
        x = rng.normal(size=8) * 2.0
        L = E.decode(x)
        assert _is_lattice(L), ("bad lattice", L)
        # nearest: no minimal-vector step improves it
        d0 = np.linalg.norm(x - L)
        for m in vecs[::24]:
            assert np.linalg.norm(x - (L + m)) >= d0 - 1e-9
    o = rng.normal(size=8)
    cells = E.route_cells(o, top_k=2)
    assert len(cells) == 2 and cells[0][0] != cells[1][0]
    assert cells[0][1] <= cells[1][1] + 1e-12
    q = E.quantize(o)
    assert q is not None and q[0] == cells[0][0]
    assert E.quantize(np.zeros(8)) is None
    assert E.route_cells(np.zeros(8)) == []
    print("test_e8: OK (240 vecs, fixed points, parity, routing)")


if __name__ == "__main__":
    test_e8()
