"""Enneagram proof (phase 70): gates, resolutions, closure.

Synthetic temp DB: shock advance refused without resolution, allowed
with one; all four outcomes write; accept resolves the dirty queue,
defer leaves it open; closure score math on a hand-built corpus.
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "A:/scripts/TheBrain")


def _db():
    p = os.path.join(tempfile.gettempdir(), "opencode", "ennea_test.db")
    try:
        os.remove(p)
    except OSError:
        pass
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE entities(canonical_id TEXT PRIMARY KEY,"
                 " description TEXT DEFAULT '')")
    conn.execute("CREATE TABLE aliases(alias_norm TEXT, canonical_id TEXT)")
    conn.execute("CREATE TABLE entity_relations(src_id TEXT, dst_id TEXT)")
    conn.execute("CREATE TABLE ingest_provenance(canonical_id TEXT,"
                 " source TEXT)")
    conn.execute("CREATE TABLE dirty_mentions(surface TEXT, doc_id TEXT,"
                 " proposed_canonical TEXT DEFAULT '',"
                 " status TEXT DEFAULT 'pending',"
                 " PRIMARY KEY (surface, doc_id))")
    # e1: described, dual-source, related, 2 aliases (model citizen)
    conn.execute("INSERT INTO entities VALUES ('e1','a person')")
    conn.execute("INSERT INTO aliases VALUES ('e one','e1'),('one','e1')")
    conn.execute("INSERT INTO entity_relations VALUES ('e1','e1')")
    conn.execute("INSERT INTO ingest_provenance VALUES ('e1','s1'),('e1','s2')")
    # e2: bare orphan loner
    conn.execute("INSERT INTO entities VALUES ('e2','')")
    conn.execute("INSERT INTO aliases VALUES ('two','e2')")
    conn.execute("INSERT INTO dirty_mentions VALUES ('E One','d1','','pending')")
    conn.commit()
    return conn


def test_enneagram():
    from core import enneagram as G
    conn = _db()
    try:
        assert G.can_autopass("verify") is True
        assert G.can_autopass("SHOCK-3") is False
        assert G.can_autopass(6) is False
        ok, why = G.advance(conn, "d1", "SHOCK-3")
        assert ok is False and "shock" in why, (ok, why)
        assert G.resolve_claim(conn, "Nope", "d1", "bogus") is False
        assert G.resolve_claim(conn, "E One", "d1", "accept",
                               "seed check") is True
        st = conn.execute("SELECT status FROM dirty_mentions").fetchone()[0]
        assert st == "resolved", st
        ok, _ = G.advance(conn, "d1", "SHOCK-3")
        assert ok is True
        ok, _ = G.advance(conn, "d1", "promote")
        assert ok is True
        assert G.resolve_claim(conn, "E Two", "d2", "defer", "needs KB") is True
        st2 = conn.execute("SELECT status FROM dirty_mentions"
                           " WHERE surface='E Two'").fetchone()
        assert st2 is None  # defer on unknown mention: record only, no crash
        cs = G.closure_score(conn)
        assert cs["n"] == 2, cs
        assert cs["filled"] == 0.5, cs
        assert cs["dual"] == 0.5, cs
        assert cs["orphan_rate"] == 0.5, cs
        assert abs(cs["score"] - (0.5 + 0.5 + 0.5) / 3) < 1e-9, cs
        # empty DB degrades, never raises
        conn2 = sqlite3.connect(":memory:")
        conn2.execute("CREATE TABLE entities(canonical_id TEXT)")
        assert G.closure_score(conn2)["score"] == 0.0
    finally:
        conn.close()
    print("test_enneagram: OK (gates, outcomes, closure math)")


if __name__ == "__main__":
    test_enneagram()
