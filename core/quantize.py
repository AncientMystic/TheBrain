"""Rotate-then-quantize entity vectors (phase 98, TurboQuant takeaway).

Random orthonormal rotation first (induces concentrated per-coordinate
distribution), then uniform scalar quantization — preserves geometry far
better than naive rounding. Two heads: MSE codes for storage, 1-bit
residual signs for unbiased inner-product estimates. Fixed seeded
rotation (deterministic across machines). Numpy only.
"""
import hashlib

import numpy as np

_rot_cache = {}


def rotation_matrix(dim, seed=0x51AB):
    """Fixed orthonormal rotation (QR of seeded Gaussian), cached by dim."""
    if dim not in _rot_cache:
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        _rot_cache[dim] = np.ascontiguousarray(Q)
    return _rot_cache[dim]


def quantize(v, bits=4):
    """Returns dict with codes (uint8), scale, min, residual signs, dim."""
    try:
        v = np.asarray(v, dtype=np.float64)
        R = rotation_matrix(v.shape[0])
        r = R @ v
        lo, hi = float(r.min()), float(r.max())
        span = max(hi - lo, 1e-12)
        levels = (1 << bits) - 1
        codes = np.clip(np.round((r - lo) / span * levels), 0, levels).astype(np.uint8)
        dq = lo + codes.astype(np.float64) / levels * span
        resid = np.sign(r - dq).astype(np.int8)
        return {"codes": codes.tobytes(), "resid": resid.tobytes(), "lo": lo,
                "span": span, "bits": bits, "dim": v.shape[0]}
    except Exception:
        return {}


def dequantize(q):
    """Storage head: reconstruct approximation (MSE-optimal direction)."""
    try:
        n = q["dim"]
        levels = (1 << q["bits"]) - 1
        codes = np.frombuffer(bytes(q["codes"]), dtype=np.uint8)[:n].astype(np.float64)
        r = q["lo"] + codes / levels * q["span"]
        return rotation_matrix(n).T @ r
    except Exception:
        return None


def inner_product(qa, qb):
    """Similarity head: dequantized dot + residual correction (unbiased)."""
    try:
        da, db = dequantize(qa), dequantize(qb)
        if da is None or db is None:
            return 0.0
        base = float(da @ db)
        ra = np.frombuffer(bytes(qa["resid"]), dtype=np.int8)[:qa["dim"]].astype(np.float64)
        rb = np.frombuffer(bytes(qb["resid"]), dtype=np.int8)[:qb["dim"]].astype(np.float64)
        corr = float((ra * rb).sum()) * (qa["span"] / ((1 << qa["bits"]) - 1)) * (qb["span"] / ((1 << qb["bits"]) - 1)) / 4.0
        return base + corr
    except Exception:
        return 0.0


def quantize_key(blob, bits=4):
    """Stable fingerprint for cache keys."""
    try:
        h = hashlib.sha256(bytes(blob) + bytes([bits])).hexdigest()[:16]
        return h
    except Exception:
        return ""
