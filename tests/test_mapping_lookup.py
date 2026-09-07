"""candidates_fn contract: exact-first, FTS fallback, capped, read-only."""
from scripts.init_mapping_db import init_mapping_db
from core.mapping_lookup import mapping_candidates_fn, lookup_display, normalize_mention


def _db(tmp_path):
    conn = init_mapping_db(str(tmp_path / "lk.db"))
    conn.execute("INSERT INTO entities (canonical_id, entity_type, type_family, display_name, shard_key, description) VALUES (?,?,?,?,?,?)",
                 ("per:marie-curie", "scientist", "person", "Marie Curie", "person:hist", "Physicist"))
    conn.execute("INSERT INTO aliases (alias_norm, canonical_id) VALUES (?,?)", ("marie curie", "per:marie-curie"))
    conn.execute("INSERT INTO aliases_fts (alias_norm, canonical_id) VALUES (?,?)", ("marie curie", "per:marie-curie"))
    conn.execute("INSERT INTO entities (canonical_id, entity_type, type_family, display_name, shard_key) VALUES (?,?,?,?,?)",
                 ("per:pierre-curie", "scientist", "person", "Pierre Curie", "person:hist"))
    conn.execute("INSERT INTO aliases (alias_norm, canonical_id) VALUES (?,?)", ("pierre curie", "per:pierre-curie"))
    conn.execute("INSERT INTO aliases_fts (alias_norm, canonical_id) VALUES (?,?)", ("pierre curie", "per:pierre-curie"))
    conn.commit()
    return conn


def test_exact_first(tmp_path):
    conn = _db(tmp_path)
    out = mapping_candidates_fn("Marie Curie!", conn=conn)
    assert out and out[0][0] == "per:marie-curie", out
    assert all(isinstance(c, tuple) and len(c) == 2 for c in out)
    conn.close()


def test_fts_fallback_and_cap(tmp_path):
    conn = _db(tmp_path)
    out = mapping_candidates_fn("curie", conn=conn, limit=1)
    assert len(out) == 1, out
    out2 = mapping_candidates_fn("", conn=conn)
    assert out2 == []
    conn.close()


def test_display_card(tmp_path):
    conn = _db(tmp_path)
    card = lookup_display(conn, "per:marie-curie")
    assert card["display_name"] == "Marie Curie" and "Physicist" in card["description"]
    assert lookup_display(conn, "nope") is None
    conn.close()


def test_normalize():
    assert normalize_mention("Marie Curie!") == "marie curie"
    assert normalize_mention("") == ""
