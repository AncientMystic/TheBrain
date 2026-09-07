"""Drain proof (phase 73): classification + apply semantics on temp DB."""
import os
import sys
import tempfile

sys.path.insert(0, "A:/scripts/TheBrain")
sys.path.insert(0, "A:/scripts/TheBrain/scripts")


def _build():
    from init_mapping_db import init_mapping_db
    from migrate_hyper_coords import migrate
    p = os.path.join(tempfile.gettempdir(), "opencode", "drain_test.db")
    try:
        os.remove(p)
    except OSError:
        pass
    conn = init_mapping_db(p)
    migrate(conn)
    conn.execute("INSERT INTO entities(canonical_id, entity_type, type_family,"
                 " display_name, shard_key) VALUES (?,?,?,?,?)",
                 ("zed", "person", "person", "Zed Zebedee", "person:global"))
    conn.execute("INSERT INTO entities(canonical_id, entity_type, type_family,"
                 " display_name, shard_key) VALUES (?,?,?,?,?)",
                 ("paris-fr", "place", "place", "Paris", "geo:eu"))
    conn.execute("INSERT INTO entities(canonical_id, entity_type, type_family,"
                 " display_name, shard_key) VALUES (?,?,?,?,?)",
                 ("paris-tx", "place", "place", "Paris", "geo:na"))
    for a, c in [("zed zebedee", "zed"), ("paris", "paris-fr"),
                 ("paris", "paris-tx")]:
        conn.execute("INSERT INTO aliases(alias_norm, canonical_id)"
                     " VALUES (?,?)", (a, c))
        conn.execute("INSERT INTO aliases_fts(alias_norm, canonical_id)"
                     " VALUES (?,?)", (a, c))
    for s in ["Zed Zebedee", "Xyzzy Qqq", "Paris"]:
        conn.execute("INSERT INTO dirty_mentions(surface, doc_id, status)"
                     " VALUES (?,?,'pending')", (s, "d9"))
    conn.commit()
    return conn


def test_drain():
    sys.path.insert(0, "A:/scripts/TheBrain/scripts")
    from drain_dirty import classify, drain
    from core.enneagram import init_tables
    conn = _build()
    try:
        assert init_tables(conn) is True
        assert classify(conn, "Zed Zebedee")[0] == "accept"
        assert classify(conn, "Xyzzy Qqq")[0] == "defer"
        assert classify(conn, "Paris")[0] == "review"
        counts = drain(conn, apply=False)
        assert counts == {"accept": 1, "defer": 1, "review": 1,
                          "applied": 0}, counts
        assert conn.execute("SELECT COUNT(*) FROM claim_resolutions"
                            ).fetchone()[0] == 0
        counts = drain(conn, apply=True)
        assert counts["applied"] == 1, counts
        st = dict(conn.execute("SELECT surface, status FROM dirty_mentions"
                               ).fetchall())
        assert st["Zed Zebedee"] == "resolved", st
        assert st["Paris"] == "pending", st
        res = conn.execute("SELECT outcome FROM claim_resolutions"
                           " WHERE surface='Zed Zebedee'").fetchone()
        assert res and res[0] == "accept", res
    finally:
        conn.close()
    print("test_drain: OK (classify, dry-run, apply-once)")


if __name__ == "__main__":
    test_drain()
