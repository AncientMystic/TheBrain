"""E8 routing lattice (phase 68): discrete quantization layer, NOT an embedding.

Each unit-scaled octonion snaps to its nearest E8 lattice point (exact
two-coset decode: D8 vs D8+half); the lattice coordinates are the routing
key in entities.e8_key. Lookup ranks the decoded cell plus its 240
minimal-vector neighbors by exact E8 distance and searches the top-2
distinct cells present in the DB — bounded lookup, same quality (ranking
inside cells is untouched; widen to top-3 or bypass on any regression).
Numpy only. Never raises from public helpers (None/[] on bad input).
"""
import itertools

import numpy as np

SCALE = 2.0  # unit octonion * SCALE lands on populated E8 shells


def _min_vectors():
    """The 240 E8 minimal vectors (norm sqrt(2)): 112 integer + 128 half."""
    vecs = []
    for i, j in itertools.combinations(range(8), 2):
        for s1 in (1.0, -1.0):
            for s2 in (1.0, -1.0):
                v = np.zeros(8)
                v[i], v[j] = s1, s2
                vecs.append(v)
    for bits in itertools.product((0.5, -0.5), repeat=8):
        if sum(1 for b in bits if b < 0) % 2 == 0:
            vecs.append(np.array(bits))
    return vecs


_MIN_VECS = None


def min_vectors():
    global _MIN_VECS
    if _MIN_VECS is None:
        _MIN_VECS = _min_vectors()
    return _MIN_VECS


def _best_even(base, is_half):
    """Closest vector to base with E8 parity (brute-force over 17 options).

    D8: integer coords, even sum. Coset: half-integer coords with
    (coord - 1/2) summing even. Obviously-correct by construction.
    """
    r = np.floor(np.asarray(base, dtype=np.float64) + 0.5)
    if is_half:
        r = r + 0.5
    cands = [r]
    for i in range(8):
        for d in (1.0, -1.0):
            c = r.copy()
            c[i] += d
            cands.append(c)
    best, bd = None, None
    for c in cands:
        if is_half:
            ok = (int(round((c - 0.5).sum())) % 2 == 0)
        else:
            ok = (int(round(c.sum())) % 2 == 0)
        if not ok:
            continue
        d = float(np.sum((c - base) ** 2))
        if bd is None or d < bd:
            best, bd = c, d
    return best, bd


def decode(x):
    """Nearest E8 lattice point to x (exact two-coset decode)."""
    x = np.asarray(x, dtype=np.float64)
    c0, d0 = _best_even(x, False)
    # Coset D8+1/2: even-sum integer vector best approximating (x-0.5),
    # shifted back by +0.5 — exactly the half-lattice parity condition.
    c1, _d = _best_even(x - 0.5, False)
    c1 = c1 + 0.5
    d1 = float(np.sum((c1 - x) ** 2))
    return c0 if d0 <= d1 else c1


def key_of(lattice_pt):
    """Stable string key for lattice coordinates (ints and .5s)."""
    parts = []
    for v in np.asarray(lattice_pt, dtype=np.float64):
        r = round(float(v) * 2.0)
        parts.append(str(r // 2) if r % 2 == 0 else f"{r / 2.0:.1f}")
    return "e8:" + ",".join(parts)


def quantize(o):
    """Octonion -> (e8_key, cell_distance). None on bad input."""
    try:
        v = np.asarray(o, dtype=np.float64)
        n = float(np.linalg.norm(v))
        if not (n > 0) or v.shape != (8,):
            return None
        q = v / n * SCALE
        L = decode(q)
        return key_of(L), float(np.linalg.norm(q - L))
    except Exception:
        return None


def route_cells(o, top_k=2):
    """Top-k routing cells for a query octonion: decoded cell + 240 minimal
    neighbors ranked by exact distance. Returns [(key, dist)]."""
    try:
        v = np.asarray(o, dtype=np.float64)
        n = float(np.linalg.norm(v))
        if not (n > 0) or v.shape != (8,):
            return []
        q = v / n * SCALE
        L0 = decode(q)
        cands = [(L0, float(np.linalg.norm(q - L0)))]
        for m in min_vectors():
            L = L0 + m
            cands.append((L, float(np.linalg.norm(q - L))))
        cands.sort(key=lambda t: t[1])
        out, seen = [], set()
        for L, d in cands:
            k = key_of(L)
            if k not in seen:
                seen.add(k)
                out.append((k, d))
            if len(out) >= max(int(top_k), 1):
                break
        return out
    except Exception:
        return []


def route_entities(o, conn, top_cells=2, limit=50):
    """Bounded candidate lookup: canonical_ids from the top routing cells.

    Read-only. Falls back to [] when cells are empty (caller uses the
    alias pool instead — bypass discipline from the plan).
    """
    try:
        cells = route_cells(o, top_k=top_cells)
        if not cells:
            return []
        out = []
        for k, d in cells:
            try:
                for row in conn.execute(
                        "SELECT canonical_id FROM entities WHERE e8_key=?"
                        " LIMIT ?", (k, int(limit))):
                    out.append((row[0], d))
                    if len(out) >= int(limit):
                        return out
            except Exception:
                continue
        return out
    except Exception:
        return []
