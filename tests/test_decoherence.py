"""Decoherence proof (phase 69): gating behavior on synthetic mentions.

- 'Paris' + 'Texas' with (paris-fr, paris-tx) + (texas) candidates:
  coherence picks paris-tx, D high, no review.
- Near-tied candidates for one mention: ambiguity -> C collapses ->
  review True (shock-gate routing).
- Gamma monotonicity: higher gamma reviews at least as much.
- Backward compat: collective_link still returns the bare choice dict.
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

import numpy as np


def _h(v):
    from core.hyperbolic import exp_map
    return np.asarray(exp_map(np.asarray(v, dtype=np.float32)))


def test_decoherence():
    from core.entity_linking import (collective_link, decoherence_link,
                                     decoherence_report)
    rng = np.random.default_rng(9)
    dim = 4  # low dim + moderate vectors stay mid-ball (no saturation)
    from core.hyperbolic import ensure_hyperbolic as _eh
    e_fr = _h(np.array([0.5, 0.2, -0.3, 0.1], dtype=np.float32))
    e_tx = _h(np.array([-0.4, 0.9, 0.2, -0.6], dtype=np.float32))
    # near-neighbor WITHOUT double exp_map (saturation would fake distance)
    e_texas = _eh((e_tx.astype(np.float64)
                   + rng.normal(scale=0.05, size=dim)).astype(np.float32),
                  space='hyperbolic')
    q_paris = (e_fr + e_tx) / 2.0  # ambiguous surface form
    q_texas = e_texas + rng.normal(scale=0.01, size=dim)

    def cands(m):
        if m == "Paris":
            return [("paris-fr", e_fr), ("paris-tx", e_tx)]
        if m == "Texas":
            return [("texas", e_texas)]
        return []

    def emb(t):
        return {"Paris": q_paris, "Texas": q_texas}[t]

    base = collective_link(["Paris", "Texas"], cands, embed_fn=emb)
    assert isinstance(base, dict) and base[1] == "texas", base
    choice, rep = decoherence_link(["Paris", "Texas"], cands, embed_fn=emb)
    assert choice == base, (choice, base)
    assert rep[1]["review"] is False and rep[1]["D"] > 0.5, rep[1]
    # joint coherence: Paris resolves to the Texas-compatible candidate
    assert choice[0] == "paris-tx", choice

    # near-tie -> review
    e_a = _h(rng.normal(size=dim))
    e_b = (e_a + rng.normal(scale=1e-4, size=dim)).astype(np.float32)
    from core.hyperbolic import ensure_hyperbolic
    e_b = ensure_hyperbolic(e_b, space='hyperbolic')
    q = (e_a.astype(np.float64) + e_b.astype(np.float64)) / 2.0

    def cands2(m):
        return [("a", e_a), ("b", e_b)]

    _, rep2 = decoherence_link(["X"], cands2, embed_fn=lambda t: q,
                               gamma=5.0)
    assert rep2[0]["review"] is True, rep2[0]
    assert rep2[0]["reason"] == "ambiguous", rep2[0]

    # gamma monotonicity on the tie case
    _, lo = decoherence_link(["X"], cands2, embed_fn=lambda t: q, gamma=0.01)
    _, hi = decoherence_link(["X"], cands2, embed_fn=lambda t: q, gamma=50.0)
    assert hi[0]["C"] <= lo[0]["C"], (lo[0], hi[0])

    # empty / unresolved degrades to review, never raises
    _, rep3 = decoherence_link([], cands, embed_fn=emb)
    assert rep3 == {}
    _, rep4 = decoherence_link(["Nobody"], lambda m: [], embed_fn=emb)
    assert rep4[0]["review"] is True, rep4
    print("test_decoherence: OK (gating, tie-review, gamma, degradation)")


if __name__ == "__main__":
    test_decoherence()
