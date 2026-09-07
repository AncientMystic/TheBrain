"""S-scoring proof (phase 65): Marie Curie must beat Curie, Texas.

Synthetic temp DB: two 'curie' candidates, query emb near Marie's, Paris
map position + person-shard context. Asserts correct winner, S in [0,1],
part weights present, graceful degradation (no emb / empty mention).
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "A:/scripts/TheBrain")
sys.path.insert(0, "A:/scripts/TheBrain/scripts")

import numpy as np


def _build():
    from init_mapping_db import init_mapping_db
    from migrate_hyper_coords import migrate
    from core.hyperbolic import exp_map
    p = os.path.join(tempfile.gettempdir(), "opencode", "score_test.db")
    try:
        os.remove(p)
    except OSError:
        pass
    conn = init_mapping_db(p)
    migrate(conn)
    rng = np.random.default_rng(11)
    dim = 16
    v_marie = rng.normal(size=dim).astype(np.float32)
    v_tx = rng.normal(size=dim).astype(np.float32) + 5.0
    e_marie = np.asarray(exp_map(v_marie), dtype=np.float32)
    e_tx = np.asarray(exp_map(v_tx), dtype=np.float32)
    conn.execute(
        "INSERT INTO entities(canonical_id, entity_type, type_family,"
        " display_name, shard_key, emb, lat_r, lon_r, depth)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        ("marie", "person", "person", "Marie Curie", "person:hist",
         sqlite3.Binary(e_marie.tobytes()), 48.85, 2.35, 0.9))
    conn.execute(
        "INSERT INTO entities(canonical_id, entity_type, type_family,"
        " display_name, shard_key, emb, lat_r, lon_r, depth)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        ("curie-tx", "place", "place", "Curie, Texas", "geo:na",
         sqlite3.Binary(e_tx.tobytes()), 32.9, -96.0, 0.9))
    for a, c in [("marie curie", "marie"), ("curie", "marie"),
                 ("curie", "curie-tx")]:
        conn.execute("INSERT INTO aliases(alias_norm, canonical_id)"
                     " VALUES (?,?)", (a, c))
        conn.execute("INSERT INTO aliases_fts(alias_norm, canonical_id)"
                     " VALUES (?,?)", (a, c))
    conn.commit()
    q = np.asarray(exp_map(v_marie + rng.normal(
        scale=0.01, size=dim).astype(np.float32)), dtype=np.float64)
    return conn, q


def test_curie_in_paris():
    from core.mapping_lookup import rank_candidates_scored
    conn, q = _build()
    try:
        res = rank_candidates_scored(
            "Curie in Paris", mention_emb=q, limit=5,
            shard_weights={"person:hist": 1.0, "geo:na": 0.2},
            qlat=48.85, qlon=2.35, conn=conn)
        assert len(res) == 2, res
        assert res[0][0] == "marie", res
        assert res[0][1] > res[1][1], res
        for cid, s, parts in res:
            assert 0.0 <= s <= 1.0, (cid, s)
        assert set(("hyp", "plane", "shard")) <= set(res[0][2]), res[0][2]
        # Degradation: no emb, no map, no shards -> neutral 0.5, both kept
        res2 = rank_candidates_scored("curie", limit=5, conn=conn)
        assert len(res2) == 2 and all(s == 0.5 for _, s, _ in res2), res2
        # Empty mention -> []
        assert rank_candidates_scored("!!!", conn=conn) == []
    finally:
        conn.close()
    print("test_hyper_score: OK (marie wins, parts present, degrades clean)")


if __name__ == "__main__":
    test_curie_in_paris()
