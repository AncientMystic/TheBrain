"""Derive hyper-coordinates per shard (phase 63 candidate).

For each shard: centroid = frechet_mean(embs); tangents = log_mu(centroid,
each); u1/u2 = top-2 PCA directions of tangents (stored per shard);
lat/lon = degree-scaled projections (float twins + 18dp identifier
strings); depth = ||emb||. Rows with NULL shard_key derive under shard
'global'. Resume-safe: skips shards already in shards table and rows with
lat_r set, unless --redo. Run --selftest for synthetic proof.

Usage: derive_coords.py [--redo] [--selftest] [db-path]
"""
import sqlite3
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def derive_shard(conn, shard, rows, exp_dim, redo=False):
    import numpy as _np
    from core.hyperbolic import (frechet_mean, log_mu, hyperbolic_distance,
                                 ensure_hyperbolic)
    ids, vecs = [], []
    for r in rows:
        try:
            v = _np.frombuffer(bytes(r[1]), dtype=_np.float32)
            if len(v) != exp_dim:
                continue
            ids.append(r[0])
            vecs.append(ensure_hyperbolic(v).astype(_np.float64))
        except Exception:
            continue
    if not ids:
        return 0
    X = _np.stack(vecs)
    mu = _np.asarray(frechet_mean(X), dtype=_np.float64)
    T = _np.stack([_np.asarray(log_mu(mu, x), dtype=_np.float64) for x in X])
    # Single-point shards (and identical twins) yield NaN tangents (0/0 in
    # the log map); by definition a point at its own centroid has zero
    # tangent. SQLite stores NaN as NULL, which orphaned such rows.
    T = _np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    Tc = T - T.mean(axis=0)
    try:
        _, _, Vt = _np.linalg.svd(Tc, full_matrices=False)
        u1, u2 = Vt[0], Vt[1] if Vt.shape[0] > 1 else _np.zeros_like(Vt[0])
    except Exception:
        # Degenerate shard (tiny/identical tangents): deterministic
        # orthonormal fallback seeded by shard key (stable across redos).
        import hashlib as _hl
        _seed = int.from_bytes(_hl.sha256(shard.encode()).digest()[:8], "big")
        _rng = _np.random.default_rng(_seed)
        _Q, _ = _np.linalg.qr(_rng.normal(size=(T.shape[1], 2)))
        u1, u2 = _Q[:, 0].copy(), _Q[:, 1].copy()
    p1, p2 = T @ u1, T @ u2
    # Per-axis scales: each axis uses its full range, zero clipping.
    m1 = float(_np.abs(p1).max(initial=0.0))
    m2 = float(_np.abs(p2).max(initial=0.0))
    k1, k2 = 90.0 / max(m1, 1e-12), 180.0 / max(m2, 1e-12)
    dmax = float(max(m1, m2, 1e-12))
    lat = _np.clip(k1 * p1, -90.0, 90.0)
    lon = _np.clip(k2 * p2, -180.0, 180.0)
    # Depth: rank-normalized centroid distance within the shard. Raw
    # Poincare norms saturate (mxbai magnitudes) so absolute depth is
    # uninformative; the rank preserves specificity order and uses the
    # full [0,1) range deterministically (ties share averaged order).
    tn = _np.linalg.norm(T, axis=1)
    order = _np.argsort(_np.argsort(tn, kind="stable"), kind="stable")
    n1 = max(len(ids) - 1, 1)
    depth = order.astype(_np.float64) / n1 * 0.999999
    radius = float(max(hyperbolic_distance(mu, x) for x in X))
    conn.execute(
        "INSERT OR REPLACE INTO shards(shard_key, centroid, u1, u2, k, k2,"
        " dmax, radius, affinity_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (shard, sqlite3.Binary(mu.astype(_np.float32).tobytes()),
         sqlite3.Binary(u1.astype(_np.float32).tobytes()),
         sqlite3.Binary(u2.astype(_np.float32).tobytes()),
         k1, k2, dmax, radius, "{}"))
    for cid, la, lo, d in zip(ids, lat, lon, depth):
        conn.execute(
            "UPDATE entities SET lat_r=?, lon_r=?, lat_str=?, lon_str=?,"
            " depth=?, shard_key=? WHERE canonical_id=?",
            (float(la), float(lo), f"{la:.18f}", f"{lo:.18f}", float(d),
             shard, cid))
    conn.commit()
    return len(ids)


def derive_all(conn, redo=False):
    from core import db as _db  # noqa: keep pool semantics intact
    import config as _cfg
    exp_dim = int(getattr(_cfg, "EMBEDDING_DIM", 1024))
    shards = [r[0] for r in conn.execute(
        "SELECT DISTINCT COALESCE(shard_key,'global') FROM entities"
        " WHERE emb IS NOT NULL")]
    if not redo:
        have = {r[0] for r in conn.execute("SELECT shard_key FROM shards")}
        shards = [s for s in shards if s not in have]
    n = 0
    for s in shards:
        rows = conn.execute(
            "SELECT canonical_id, emb FROM entities WHERE emb IS NOT NULL"
            " AND lat_r IS NULL AND COALESCE(shard_key,'global')=?", (s,)).fetchall() \
            if not redo else conn.execute(
            "SELECT canonical_id, emb FROM entities WHERE emb IS NOT NULL"
            " AND COALESCE(shard_key,'global')=?", (s,)).fetchall()
        print(f"derive_coords: shard {s} ({len(rows)} rows)...", flush=True)
        got = derive_shard(conn, s, rows, exp_dim, redo)
        print(f"derive_coords: shard {s} done ({got} rows)", flush=True)
        n += got
    return n


def selftest():
    import numpy as _np
    import tempfile, os
    sys.path.insert(0, "A:/scripts/TheBrain/scripts")
    from init_mapping_db import init_mapping_db
    from migrate_hyper_coords import migrate
    p = os.path.join(tempfile.gettempdir(), "opencode", "coord_test.db")
    try:
        os.remove(p)
    except OSError:
        pass
    conn = init_mapping_db(p)
    migrate(conn)
    rng = _np.random.default_rng(7)
    for i in range(40):
        v = rng.normal(size=32).astype(_np.float32)
        conn.execute(
            "INSERT INTO entities(canonical_id, entity_type, type_family,"
            " display_name, shard_key, emb) VALUES (?,?,?,?,?,?)",
            (f"t{i}", "person", "person", f"Test {i}", "global",
             sqlite3.Binary(v.tobytes())))
    conn.commit()
    import config as _cfg
    _old, _cfg.EMBEDDING_DIM = _cfg.EMBEDDING_DIM, 32
    try:
        n = derive_all(conn)
    finally:
        _cfg.EMBEDDING_DIM = _old
    assert n == 40, n
    r = conn.execute("SELECT lat_r, lon_r, lat_str, lon_str, depth,"
                     " shard_key FROM entities").fetchall()
    assert all(x[0] is not None and -90 <= x[0] <= 90 for x in r)
    assert all(x[1] is not None and -180 <= x[1] <= 180 for x in r)
    assert all(len(x[2].split(".")[1]) == 18 for x in r)
    assert all(0.0 <= x[4] < 1.0 for x in r)
    assert all(x[5] == "global" for x in r)
    s = conn.execute("SELECT k, k2, dmax, radius FROM shards").fetchone()
    assert s[0] > 0 and s[1] > 0 and s[2] > 0 and s[3] >= 0
    # rank depth uses the full range even for saturated norms
    ds = [x[0] for x in conn.execute("SELECT depth FROM entities")]
    assert len(set(ds)) == 40 and min(ds) >= 0.0 and max(ds) <= 0.999999
    # determinism: rerun with redo, strings must be identical
    before = [tuple(x) for x in conn.execute(
        "SELECT lat_str, lon_str FROM entities ORDER BY canonical_id")]
    _cfg.EMBEDDING_DIM = 32
    try:
        derive_all(conn, redo=True)
    finally:
        _cfg.EMBEDDING_DIM = _old
    after = [tuple(x) for x in conn.execute(
        "SELECT lat_str, lon_str FROM entities ORDER BY canonical_id")]
    assert before == after, "non-deterministic coordinates"
    conn.close()
    print("derive_coords selftest: OK (40 rows, ranges, 18dp, deterministic)")
    return True


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    _redo = "--redo" in sys.argv
    _dbflag = [a.split("=", 1)[1] for a in sys.argv[1:]
               if a.startswith("--db=")]
    _dbname = _dbflag[0] if _dbflag else "mapping"
    _rest = [a for a in sys.argv[1:]
             if a != "--redo" and not a.startswith("--db")]
    if _rest:
        _conn = sqlite3.connect(_rest[0])
    else:
        from core import db as _db
        _conn = _db.db_connect(_dbname)
    _n = derive_all(_conn, redo=_redo)
    print(f"derive_coords: {_n} rows assigned")
    _conn.close()
