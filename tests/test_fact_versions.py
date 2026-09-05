"""Append-only valuations: new versions on change, same id when unchanged."""
from core.fact_versions import init_versions_table, record_version, get_history
from core import db


def _clean(fact_id):
    conn = db.db_connect("key_facts")
    try:
        conn.execute("DELETE FROM fact_versions WHERE fact_id=?", (fact_id,))
        conn.commit()
    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def test_version_then_unchanged_returns_same():
    init_versions_table()
    _clean(-101)
    fact = {"fact_id": -101, "fact_text": "Versioned claim here.", "confidence": 0.8,
            "verification_status": "verified", "truth_class": "OBSERVED",
            "verification_layers": [{"layer": "symstep", "verified": True}]}
    v1 = record_version(dict(fact), -101)
    assert isinstance(v1, int)
    v2 = record_version(dict(fact), -101)
    assert v2 == v1
    hist = get_history(-101)
    assert len(hist) == 1 and hist[0]["verification_status"] == "verified"
    _clean(-101)


def test_change_supersedes():
    init_versions_table()
    _clean(-102)
    fact = {"fact_id": -102, "fact_text": "Evolving claim.", "confidence": 0.5,
            "verification_status": "unverified"}
    v1 = record_version(dict(fact), -102)
    fact["confidence"] = 0.9
    fact["verification_status"] = "verified"
    v2 = record_version(dict(fact), -102)
    assert v2 != v1
    hist = get_history(-102)
    assert len(hist) == 2
    assert hist[0]["supersedes_version_id"] == hist[1]["version_id"]
    _clean(-102)


def test_never_raises_on_garbage():
    record_version({}, None)  # must not raise
    assert get_history(-999999) == []
