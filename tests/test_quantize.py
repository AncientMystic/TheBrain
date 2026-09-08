"""Quantize proof (phase 98): rotation helps, heads agree, deterministic."""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

import numpy as np


def test_quantize():
    from core import quantize as Q
    rng = np.random.default_rng(11)
    vecs = [rng.normal(size=128) * (1 + (i % 5)) for i in range(40)]
    # round-trip error with vs without rotation (rotation ON here; naive baseline inline)
    errs, naive = [], []
    for v in vecs:
        q = Q.quantize(v, bits=4)
        assert q and len(q["codes"]) == 128
        d = Q.dequantize(q)
        assert d is not None and len(d) == 128
        errs.append(float(np.linalg.norm(v - d) / max(np.linalg.norm(v), 1e-9)))
        lo, hi = float(v.min()), float(v.max())
        c = np.clip(np.round((v - lo) / max(hi - lo, 1e-12) * 15), 0, 15)
        dn = lo + c / 15 * max(hi - lo, 1e-12)
        naive.append(float(np.linalg.norm(v - dn) / max(np.linalg.norm(v), 1e-9)))
    import statistics
    assert statistics.mean(errs) <= statistics.mean(naive), (statistics.mean(errs), statistics.mean(naive))
    # inner-product head tracks true dots
    dots, ests = [], []
    for i in range(0, len(vecs), 2):
        a, b = vecs[i], vecs[(i + 1) % len(vecs)]
        dots.append(float(a @ b))
        ests.append(Q.inner_product(Q.quantize(a), Q.quantize(b)))
    corr = float(np.corrcoef(dots, ests)[0, 1])
    assert corr > 0.95, corr
    # determinism: same bytes twice
    assert Q.quantize(vecs[0])["codes"] == Q.quantize(vecs[0])["codes"]
    assert Q.quantize(vecs[0], bits=2)["bits"] == 2
    print(f"test_quantize: OK (rel-err {statistics.mean(errs):.3f} vs naive {statistics.mean(naive):.3f}, dot-corr {corr:.3f})")


if __name__ == "__main__":
    test_quantize()
