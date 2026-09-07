"""Octonion entity signatures (phase 67): 8-D hypersphere records.

Each entity compresses to 8 float64 components: 8 interpretable features
blended 50/50 with a FIXED seeded projection of the tangent vector, so
geometry survives compression. Uses, strictly limited (non-associative
algebra — no ML machinery touches these):
  - norm(o): principled relevance magnitude (replaces popularity tiebreak)
  - conjugate(o): contradiction probe for the verifier (negated claim
    scoring closer to evidence than the claim = flag)
  - rotate(o, k, theta): fixed named rotations in the (0,k) plane only
Fixed projection seed -> identical bytes on every machine, every run.
Numpy only, zero new dependencies. Read-only builders; never raises.
"""
import hashlib

import numpy as np

PROJECTION_SEED = 20260907
OCT_DIM = 8
BLEND_FEATURE = 0.5

_proj_cache = {}


def projection_matrix(in_dim):
    """Fixed orthonormalized random projection (8, in_dim), cached by dim."""
    if in_dim not in _proj_cache:
        rng = np.random.default_rng(PROJECTION_SEED)
        R = rng.normal(size=(OCT_DIM, in_dim))
        Q, _ = np.linalg.qr(R.T)
        _proj_cache[in_dim] = np.ascontiguousarray(Q.T[:OCT_DIM])
    return _proj_cache[in_dim]


def _type_coord(entity_type):
    """Stable [-1,1] coordinate from entity_type string (sha256, first u64)."""
    h = hashlib.sha256(str(entity_type or "").encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big")
    return (u / 2.0**64) * 2.0 - 1.0


def _log_scale(x, scale=10.0):
    try:
        return float(np.log1p(max(float(x), 0.0)) / np.log1p(scale))
    except Exception:
        return 0.0


def build_features(popularity=0, entity_type="", depth=0.0, degree=0,
                   confidence=0.5, n_sources=0, n_aliases=0):
    """8 interpretable components in roughly [-1,1]:
    [magnitude, type, temporal(0: no dates in mapping yet), spatial-depth,
    relational-degree, confidence, provenance-tier, salience]."""
    try:
        d = min(max(float(depth or 0.0), 0.0), 0.999999)
    except Exception:
        d = 0.0
    return np.array([
        _log_scale(popularity),
        _type_coord(entity_type),
        0.0,
        d * 2.0 - 1.0,
        _log_scale(degree),
        min(max(float(confidence), 0.0), 1.0) * 2.0 - 1.0,
        min(max(int(n_sources or 0), 0), 8) / 8.0 * 2.0 - 1.0,
        _log_scale(n_aliases),
    ], dtype=np.float64)


def signature(emb, popularity=0, entity_type="", depth=0.0, degree=0,
              confidence=0.5, n_sources=0, n_aliases=0):
    """Full octonion: 0.5*features + 0.5*projected unit tangent. Bytes-stable."""
    f = build_features(popularity, entity_type, depth, degree, confidence,
                       n_sources, n_aliases)
    try:
        from core.hyperbolic import log_map
        t = np.asarray(log_map(np.asarray(emb, dtype=np.float32)),
                       dtype=np.float64)
    except Exception:
        t = np.asarray(emb, dtype=np.float64)
    n = float(np.linalg.norm(t))
    t = t / n if n > 0 else np.zeros_like(t)
    R = projection_matrix(t.shape[0])
    g = R @ t
    gn = float(np.linalg.norm(g))
    if gn > 0:
        g = g / gn
    return BLEND_FEATURE * f + (1.0 - BLEND_FEATURE) * g


def norm(o):
    """Relevance magnitude."""
    return float(np.linalg.norm(np.asarray(o, dtype=np.float64)))


def conjugate(o):
    """Octonion conjugate: negate the 7 imaginary parts."""
    c = np.asarray(o, dtype=np.float64).copy()
    c[1:] *= -1.0
    return c


def rotate(o, k, theta):
    """Fixed rotation by theta in the (real, k-th imaginary) plane (1<=k<=7).

    Norm-preserving by construction. Only caller-approved (k, theta) pairs
    (named rotations like recency/locality) may be used downstream.
    """
    if not (1 <= int(k) <= 7):
        raise ValueError("k must be 1..7")
    v = np.asarray(o, dtype=np.float64).copy()
    c, s = float(np.cos(theta)), float(np.sin(theta))
    a, b = v[0], v[int(k)]
    v[0], v[int(k)] = c * a - s * b, s * a + c * b
    return v


def contradiction_probe(o_claim, o_evidence):
    """Margin > 0 means the NEGATED claim sits closer to the evidence.

    Returns (margin, flag): flag True when conjugation improves the fit,
    i.e. the verifier should challenge rather than accept the claim.
    """
    try:
        a = np.asarray(o_claim, dtype=np.float64)
        b = np.asarray(o_evidence, dtype=np.float64)
        d_direct = float(np.linalg.norm(a - b))
        d_conj = float(np.linalg.norm(conjugate(a) - b))
        margin = d_direct - d_conj
        return margin, bool(margin > 0)
    except Exception:
        return 0.0, False


def to_blob(o):
    import sqlite3
    return sqlite3.Binary(np.asarray(o, dtype=np.float64).tobytes())


def from_blob(blob):
    a = np.frombuffer(bytes(blob), dtype=np.float64)
    if len(a) != OCT_DIM:
        raise ValueError(f"oct8 dim {len(a)} != {OCT_DIM}")
    return a.copy()
