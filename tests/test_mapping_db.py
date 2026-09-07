"""P2 mapping DB: schema creation, alias round-trip, terms import."""
import sqlite3
from scripts.init_mapping_db import init_mapping_db, import_disambiguation_notes


def _tmp(tmp_path):
    p = tmp_path / "mapping_test.db"
    conn = init_mapping_db(str(p))
    return conn


def test_schema_tables(tmp_path):
    conn = _tmp(tmp_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','virtual table')")}
    for t in ("entities", "aliases", "aliases_fts", "entity_relations",
              "shard_centroids", "ingest_provenance", "dirty_mentions", "terms"):
        assert t in tables, f"missing table {t}"
    conn.close()


def test_alias_lookup_roundtrip(tmp_path):
    conn = _tmp(tmp_path)
    conn.execute("INSERT INTO entities (canonical_id, entity_type, type_family, display_name, shard_key, description) VALUES (?,?,?,?,?,?)",
                 ("geo:paris-fr", "city", "geo", "Paris", "geo:eu", "Capital of France"))
    conn.execute("INSERT INTO aliases (alias_norm, canonical_id) VALUES (?,?)", ("paris", "geo:paris-fr"))
    conn.execute("INSERT INTO aliases_fts (alias_norm, canonical_id) VALUES (?,?)", ("paris", "geo:paris-fr"))
    conn.commit()
    row = conn.execute("SELECT canonical_id FROM aliases WHERE alias_norm=?", ("paris",)).fetchone()
    assert row[0] == "geo:paris-fr"
    fts = conn.execute("SELECT canonical_id FROM aliases_fts WHERE aliases_fts MATCH ?", ('"paris"*',)).fetchone()
    assert fts[0] == "geo:paris-fr"
    conn.close()


def test_terms_import(tmp_path):
    conn = _tmp(tmp_path)
    n = import_disambiguation_notes(conn)
    assert n >= 100, f"expected 100+ terms, got {n}"
    row = conn.execute("SELECT risk, action FROM terms WHERE term='georgia'").fetchone()
    assert row[0] == "high"
    # Idempotent re-import keeps the same count.
    n2 = import_disambiguation_notes(conn)
    total = conn.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
    assert n2 == n and total == n
    conn.close()


def test_shard_routing_keys(tmp_path):
    conn = _tmp(tmp_path)
    conn.execute("INSERT INTO entities (canonical_id, entity_type, type_family, display_name, shard_key) VALUES (?,?,?,?,?)",
                 ("per:marie-curie", "scientist", "person", "Marie Curie", "person:global"))
    got = conn.execute("SELECT shard_key FROM entities WHERE canonical_id=?", ("per:marie-curie",)).fetchone()[0]
    assert got == "person:global"
    conn.close()
