"""Canon lifecycle: ordering guards, idempotency, evidence rules."""
from logic import canon as _canon


def _fresh_id():
    import time
    _fresh_id.n = getattr(_fresh_id, "n", 0) + 1
    return -1000000 - int(time.time()) % 100000 - _fresh_id.n


def _make_module(name):
    from core import db
    conn = db.db_connect("logic")
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO logic_modules (name, category, summary, keywords, content) VALUES (?,?,?,?,?)",
                    (name, "reasoning", "s", "[]", "c"))
        lid = int(cur.lastrowid)
        conn.commit()
        return lid
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _wipe(lid):
    from core import db
    for table, col in (("logic_canon_promotions", "logic_id"), ("logic_examples", "logic_id"),
                       ("logic_modules", "logic_id")):
        try:
            conn = db.db_connect("logic")
            try:
                conn.execute(f"DELETE FROM {table} WHERE {col}=?", (lid,))
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        except Exception:
            pass


def test_ordering_and_idempotency():
    lid = _make_module(f"canon-order-{_fresh_id()}")
    try:
        assert _canon.stage_of(lid) == "candidate"
        ok, _ = _canon.promote(lid, "canonical", approved=True)
        assert not ok  # validated first
        ok, _ = _canon.promote(lid, "validated")
        assert not ok  # no evidence yet
        from core import db
        conn = db.db_connect("logic")
        try:
            conn.execute("INSERT INTO logic_examples (logic_id, input_text, output_text) VALUES (?,?,?)",
                         (lid, "in", "out"))
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        ok, _ = _canon.promote(lid, "validated")
        assert ok and _canon.stage_of(lid) == "validated"
        ok, _ = _canon.promote(lid, "canonical")
        assert not ok  # needs approval flag
        ok, _ = _canon.promote(lid, "canonical", approved=True)
        assert ok and _canon.stage_of(lid) == "canonical"
        ok, _ = _canon.promote(lid, "validated")
        assert ok  # idempotent re-promotion to held stage
    finally:
        _wipe(lid)


def test_bad_stage_rejected():
    ok, _ = _canon.promote(-999999, "bogus")
    assert not ok
