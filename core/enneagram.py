"""Enneagram process records (phase 70): governed extraction pipeline.

Nine-stage pipeline with two mandatory shock gates (positions 3 and 6)
that auto-flow cannot self-pass — they require a human/external-KB
resolution. Law-of-Three: no canonical promotion without an explicit
reconciling record (accept/merge/split/defer) on the dirty queue.
Closure audit: filled slots, dual provenance, orphan rate as a single
gate score. SQLite-backed, idempotent table init, never raises.
"""
import time

STAGES = ("ingest", "extract", "candidate", "SHOCK-3", "verify",
          "evidence", "SHOCK-6", "promote", "audit")
SHOCKS = (3, 6)
OUTCOMES = ("accept", "merge", "split", "defer")


def init_tables(conn):
    """Idempotent: claim_resolutions + process_states."""
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS claim_resolutions("
            " surface TEXT NOT NULL, doc_id TEXT NOT NULL,"
            " outcome TEXT NOT NULL, notes TEXT DEFAULT '',"
            " decided_at INT DEFAULT 0,"
            " PRIMARY KEY (surface, doc_id))")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS process_states("
            " doc_id TEXT PRIMARY KEY, position INT DEFAULT 0,"
            " updated_at INT DEFAULT 0)")
        conn.commit()
        return True
    except Exception:
        return False


def can_autopass(stage):
    """Shock stages (3, 6) refuse auto-passage; all others pass."""
    try:
        pos = STAGES.index(stage) if isinstance(stage, str) else int(stage)
        return pos not in SHOCKS
    except Exception:
        return False


def advance(conn, doc_id, to_stage):
    """Move a doc's process position; shock stages need a resolution.

    Returns (ok, reason). A shock advance requires >=1 claim_resolution
    for the doc (the external input the shock point demands). Never raises.
    """
    try:
        init_tables(conn)
        pos = STAGES.index(to_stage) if isinstance(to_stage, str) else int(to_stage)
        if pos in SHOCKS:
            n = conn.execute("SELECT COUNT(*) FROM claim_resolutions"
                             " WHERE doc_id=?", (doc_id,)).fetchone()[0]
            if not n:
                return False, "shock gate: no resolution for doc"
        conn.execute("INSERT OR REPLACE INTO process_states"
                     " (doc_id, position, updated_at) VALUES (?,?,?)",
                     (doc_id, pos, int(time.time())))
        conn.commit()
        return True, "ok"
    except Exception as e:
        return False, str(e)[:120]


def resolve_claim(conn, surface, doc_id, outcome, notes=""):
    """Write a Law-of-Three reconciling record; flips dirty_mentions too.

    outcome in accept/merge/split/defer. accept/merge mark the queued
    mention resolved; split/defer leave it open with the outcome noted.
    Returns True on write. Never raises.
    """
    try:
        if outcome not in OUTCOMES:
            return False
        if not (surface or "").strip() or not (doc_id or "").strip():
            return False
        init_tables(conn)
        conn.execute("INSERT OR REPLACE INTO claim_resolutions"
                     " (surface, doc_id, outcome, notes, decided_at)"
                     " VALUES (?,?,?,?,?)",
                     (surface.strip(), doc_id.strip(), outcome,
                      notes or "", int(time.time())))
        if outcome in ("accept", "merge"):
            try:
                conn.execute("UPDATE dirty_mentions SET status='resolved'"
                             " WHERE surface=? AND doc_id=?",
                             (surface.strip(), doc_id.strip()))
            except Exception:
                pass
        else:
            try:
                conn.execute("UPDATE dirty_mentions SET status=?"
                             " WHERE surface=? AND doc_id=?",
                             (f"open-{outcome}", surface.strip(),
                              doc_id.strip()))
            except Exception:
                pass
        conn.commit()
        return True
    except Exception:
        return False


def closure_score(conn):
    """Audit composite in [0,1]: filled slots, dual provenance, orphans.

    filled: entities with non-empty description. dual: entities with >=2
    distinct provenance sources. orphan: entities with no relations and
    a single alias (unverifiable loners). score = (filled + dual +
    (1 - orphan_rate)) / 3. Read-only. Never raises.
    """
    out = {"filled": 0.0, "dual": 0.0, "orphan_rate": 0.0, "score": 0.0,
           "n": 0}
    try:
        n = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        if not n:
            return out
        out["n"] = n
        try:
            f = conn.execute("SELECT COUNT(*) FROM entities WHERE"
                             " description IS NOT NULL AND description != ''"
                             ).fetchone()[0]
            out["filled"] = f / n
        except Exception:
            pass
        try:
            d = conn.execute("SELECT COUNT(*) FROM (SELECT canonical_id"
                             " FROM ingest_provenance GROUP BY canonical_id"
                             " HAVING COUNT(DISTINCT source) >= 2)").fetchone()[0]
            out["dual"] = d / n
        except Exception:
            pass
        try:
            o = conn.execute(
                "SELECT COUNT(*) FROM entities e WHERE NOT EXISTS"
                " (SELECT 1 FROM entity_relations r WHERE r.src_id="
                "e.canonical_id OR r.dst_id=e.canonical_id)"
                " AND (SELECT COUNT(*) FROM aliases a WHERE a.canonical_id="
                "e.canonical_id) <= 1").fetchone()[0]
            out["orphan_rate"] = o / n
        except Exception:
            pass
        out["score"] = (out["filled"] + out["dual"]
                        + (1.0 - out["orphan_rate"])) / 3.0
        return out
    except Exception:
        return out
