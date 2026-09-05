"""
Regime-cube audit + trajectory diagnostics (generic, no doc-specific thresholds).

Computes marginal activation rates, joint co-occurrence, loading fingerprints,
saturation index, trajectory length, active-set stability. Used for post-training
audit and nightly monitoring. Thresholds via config, env-overridable.
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)


def activation_profile(gate, features):
    """Return [wtop, wgap, wups] in [0,1]^3 for a feature vector.

    Currently only top-band forward is implemented in PrimeEvenGate;
    gap/ups use same logistic form when loadings present, else 0.5 neutral.
    Generic, preserves gate semantics.
    """
    try:
        wtop = float(gate.forward(features))
    except Exception:
        wtop = 0.5
    # Gap / unitary approximations from gamma/delta when available
    try:
        feats = np.asarray(features, dtype=np.float32)
        # gamma: intercept + 21 gaps (features[22:43] when 44-dim)
        if hasattr(gate, "gamma") and len(feats) >= 43:
            z = float(gate.gamma[0] + np.dot(gate.gamma[1:], feats[22:43]))
            wgap = float(1.0 / (1.0 + np.exp(-z)))
        else:
            wgap = 0.5
    except Exception:
        wgap = 0.5
    try:
        if hasattr(gate, "delta") and len(feats) >= 22:
            # unitary-coupled uses first 22 + phase last when present
            phase = float(feats[-1]) if len(feats) else 0.0
            z = float(gate.delta[0] + np.dot(gate.delta[1:], feats[:22]) * (0.5 + 0.5 * phase))
            wups = float(1.0 / (1.0 + np.exp(-z)))
        else:
            wups = 0.5
    except Exception:
        wups = 0.5
    return [wtop, wgap, wups]


def marginal_rates(profiles, tau=0.5):
    import numpy as _np
    P = _np.asarray(profiles, dtype=np.float32)
    if len(P) == 0:
        return [0.0, 0.0, 0.0]
    return [float(_np.mean(P[:, g] > tau)) for g in range(3)]


def joint_matrix(profiles, tau=0.5):
    import numpy as _np
    P = _np.asarray(profiles, dtype=np.float32)
    M = [[0.0] * 3 for _ in range(3)]
    if len(P) == 0:
        return M
    for a in range(3):
        for b in range(3):
            M[a][b] = float(_np.mean((P[:, a] > tau) & (P[:, b] > tau)))
    return M


def loading_fingerprint(gate):
    import numpy as _np
    prime = {2, 3, 5, 7, 11, 13, 17, 19}
    even = set(range(2, 23, 2))
    try:
        b = _np.asarray(gate.beta[1:], dtype=np.float32)
        ps = float(sum(abs(b[i - 1]) for i in prime if i - 1 < len(b)) / (float(_np.sum(_np.abs(b))) + 1e-8))
    except Exception:
        ps = 0.0
    try:
        g = _np.asarray(gate.gamma[1:], dtype=np.float32)
        es = float(sum(abs(g[i - 1]) for i in even if i - 1 < len(g)) / (float(_np.sum(_np.abs(g))) + 1e-8))
    except Exception:
        es = 0.0
    try:
        anchor = float(gate.delta[2])
        pv = [float(gate.delta[i]) for i in prime if i < len(gate.delta)]
        avg = float(_np.mean(pv)) if pv else 0.0
        coh = float(1.0 / (1.0 + abs(anchor - avg)))
    except Exception:
        coh = 0.0
    return {"prime_support": ps, "even_support": es, "anchor_coherence": coh}


def saturation_index(profiles, hi=0.9, lo=0.1):
    import numpy as _np
    P = _np.asarray(profiles, dtype=np.float32)
    if len(P) == 0:
        return 0.0
    sat = [all((v > hi or v < lo) for v in row) for row in P]
    return float(sum(sat) / len(sat))


def trajectory_length(thetas):
    import numpy as _np
    if len(thetas) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(thetas[:-1], thetas[1:]):
        try:
            total += float(_np.linalg.norm(_np.asarray(b) - _np.asarray(a)))
        except Exception:
            continue
    try:
        straight = float(_np.linalg.norm(_np.asarray(thetas[-1]) - _np.asarray(thetas[0])))
    except Exception:
        straight = 0.0
    return total / (straight + 1e-8) if straight > 0 else total
