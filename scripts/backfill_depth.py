"""Backfill depth = ||ensure_hyperbolic(emb)|| for rows that have embeddings.

Depth semantics: 0 = generic/abstract (near origin), ->1 = specific/leaf
(near boundary). Rows without embeddings keep NULL depth until the embed
pass runs. Idempotent: recomputes every row with a non-null emb.
"""
import sqlite3
import numpy as np


def backfill(conn, batch=2000):
    from core.hyperbolic import ensure_hyperbolic
    cur = conn.cursor()
    n = 0
    for (cid, blob) in cur.execute(
            "SELECT canonical_id, emb FROM entities WHERE emb IS NOT NULL"):
        try:
            v = np.frombuffer(bytes(blob), dtype=np.float32)
            h = ensure_hyperbolic(v)
            d = float(np.linalg.norm(np.asarray(h, dtype=np.float64)))
            conn.execute("UPDATE entities SET depth=? WHERE canonical_id=?",
                         (min(max(d, 0.0), 0.999999), cid))
            n += 1
            if n % batch == 0:
                conn.commit()
        except Exception:
            continue
    conn.commit()
    return n


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        _conn = sqlite3.connect(_sys.argv[1])
    else:
        _sys.path.insert(0, "A:/scripts/TheBrain")
        from core import db as _db
        _conn = _db.db_connect("mapping")
    _n = backfill(_conn)
    _with = _conn.execute("SELECT COUNT(*) FROM entities WHERE depth IS NOT NULL").fetchone()[0]
    _tot = _conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(f"backfilled: {_n}; rows with depth: {_with}/{_tot}")
    _conn.close()
