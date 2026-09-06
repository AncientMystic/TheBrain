"""
Shared memory bus: scoped blackboard over existing memory tables.

Scopes: private (owning session only — today's behavior, default), shared
(named task runs), global (canon-grade, promotion-gated). Reads take an
allowlist defaulting to private-only, so every existing call site behaves
identically until it opts in. Promotion mirrors the canon lifecycle:
private->shared needs the producing run's id (self-service); ->global needs
explicit approval. PII redaction stays at the store edge (bus never bypasses).
"""
import logging

logger = logging.getLogger(__name__)

SCOPES = ("private", "shared", "global")


def ensure_scope_columns():
    """Additive migration: scope + provenance columns (existing rows keep working)."""
    from core import db
    conn = db.db_connect("memories")
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(memory_entries)")
        cols = {row[1] for row in cur.fetchall()}
        if "scope" not in cols:
            cur.execute("ALTER TABLE memory_entries ADD COLUMN scope TEXT DEFAULT 'private'")
        if "provenance_json" not in cols:
            cur.execute("ALTER TABLE memory_entries ADD COLUMN provenance_json TEXT")
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _candidate_ids(session_id, scopes):
    """Entry ids visible under the scope rule (single batched query)."""
    from core import db
    ensure_scope_columns()
    scopes = tuple(s for s in (scopes or ("private",)) if s in SCOPES) or ("private",)
    conn = db.db_connect("memories")
    try:
        cur = conn.cursor()
        ph = ",".join("?" for _ in scopes)
        if session_id and "private" in scopes:
            cur.execute(f"""SELECT memory_id FROM memory_entries
                            WHERE scope IN ({ph}) AND (scope != 'private' OR session_id=?)""",
                        (*scopes, session_id))
        else:
            cur.execute(f"""SELECT memory_id FROM memory_entries
                            WHERE scope IN ({ph}) AND scope != 'private'""", tuple(scopes))
        return {row[0] for row in cur.fetchall()}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def store(session_id, content, memory_type="fact", importance=0.5, scope="private",
          producer="", run_id=""):
    """Store via existing paths, then tag scope + provenance (never bypasses redact)."""
    if scope not in SCOPES:
        scope = "private"
    from memory.store import store_memory
    import json
    mid = store_memory(session_id, content, memory_type, importance)
    try:
        ensure_scope_columns()
        from core import db
        conn = db.db_connect("memories")
        try:
            conn.execute("UPDATE memory_entries SET scope=?, provenance_json=? WHERE memory_id=?",
                         (scope, json.dumps({"producer": producer, "run_id": run_id})[:2000], mid))
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
    return mid


def retrieve(query, top_k=5, session_id=None, scopes=("private",)):
    """Retrieve through existing paths, intersected with the scope rule.

    Default (private-only) delegates EXACTLY to the legacy call — byte-identical
    behavior, not approximated. The over-fetch/intersect path runs only when
    shared/global scopes are explicitly requested.
    """
    from memory.retrieve import retrieve_memories
    scopes = tuple(scopes or ("private",))
    if scopes == ("private",):
        return retrieve_memories(query, top_k=top_k, session_id=session_id)
    allowed = _candidate_ids(session_id, scopes)
    try:
        results = retrieve_memories(query, top_k=max(top_k * 5, top_k), session_id=None)
    except Exception:
        return []
    out = [r for r in results if r[1] in allowed]
    # Legacy session scoping for private rows is enforced by _candidate_ids;
    # keep the session-preference signal by stable-sorting session matches first.
    if session_id:
        try:
            from core import db
            conn = db.db_connect("memories")
            try:
                cur = conn.cursor()
                cur.execute("SELECT memory_id FROM memory_entries WHERE session_id=?", (session_id,))
                own = {row[0] for row in cur.fetchall()}
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            out.sort(key=lambda r: (0 if r[1] in own else 1))
        except Exception:
            pass
    return out[:top_k]


def promote(memory_id, scope, run_id="", approved=False):
    """Promote an entry's scope. Returns (ok, message); never raises.

    private->shared needs the producing run's id; ->global needs approval.
    Demotion appends nothing and changes nothing (no silent narrowing either):
    use a new entry for corrections, mirroring append-only valuations.
    """
    if scope not in ("shared", "global"):
        return False, f"unknown scope {scope!r}"
    if scope == "global" and not approved:
        return False, "global needs explicit human/admin approval"
    if scope == "shared" and not run_id:
        return False, "shared needs the producing run id"
    try:
        ensure_scope_columns()
        from core import db
        conn = db.db_connect("memories")
        try:
            conn.execute("UPDATE memory_entries SET scope=? WHERE memory_id=?", (scope, memory_id))
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return True, f"promoted to {scope}"
    except Exception as e:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return False, str(e)[:200]
