"""Memory bus: default parity, scope isolation, promotion guards."""
import time
from memory import bus as _bus
from memory.retrieve import retrieve_memories as _legacy_retrieve


def _sess():
    return f"bus-test-{int(time.time() * 1000) % 100000000}-{id(object()) % 1000}"


def test_default_private_matches_legacy():
    s = _sess()
    _bus.store(s, "Bus parity probe memory about harbors.", scope="private")
    assert _bus.retrieve("harbors", top_k=5, session_id=s) == \
        _legacy_retrieve("harbors", top_k=5, session_id=s)


def test_private_invisible_across_sessions():
    s1, s2 = _sess() + "a", _sess() + "b"
    _bus.store(s1, "Zebra harbor notes private.", scope="private")
    got = _bus.retrieve("zebra harbor", top_k=10, session_id=s2)
    assert all("Zebra harbor notes" not in (r[2] if len(r) > 2 else "") for r in got)


def test_shared_visible_without_session_match():
    from core import db
    s = _sess()
    mid = _bus.store(s, "Shared drydock schedule.", scope="private")
    ok, _ = _bus.promote(mid, "shared", run_id="run-1")
    assert ok
    got = _bus.retrieve("drydock schedule", top_k=10, session_id="other-session",
                        scopes=("private", "shared"))
    assert any(r[1] == mid for r in got)
    # ...but not under default private-only scope
    got2 = _bus.retrieve("drydock schedule", top_k=10, session_id="other-session")
    assert all(r[1] != mid for r in got2)
    conn = db.db_connect("memories")
    try:
        conn.execute("DELETE FROM memory_entries WHERE memory_id=?", (mid,))
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def test_promotion_guards():
    s = _sess()
    mid = _bus.store(s, "Guarded lighthouse log.", scope="private")
    try:
        ok, _ = _bus.promote(mid, "global", approved=False)
        assert not ok
        ok, _ = _bus.promote(mid, "shared", run_id="")
        assert not ok
        ok, _ = _bus.promote(mid, "bogus")
        assert not ok
        ok, _ = _bus.promote(mid, "global", approved=True)
        assert ok
    finally:
        from core import db
        conn = db.db_connect("memories")
        try:
            conn.execute("DELETE FROM memory_entries WHERE memory_id=?", (mid,))
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
