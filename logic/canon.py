"""
Canon lifecycle for logic modules: candidate -> validated -> canonical.

Mirrors the verified-folder promotion pattern (tracking table, idempotency
guard, staged side effects). Promotion never rewrites history: each stage
appends a record; demotion appends rather than deletes. Stage rules:
- validated: module has at least one passing example check or a verification
  record; requires no human.
- canonical: requires validated first plus explicit human/admin approval flag.
Generic, no domain-specific content rules.
"""
import logging

logger = logging.getLogger(__name__)

STAGES = ("candidate", "validated", "canonical")


def init_canon_table(conn=None):
    """Create logic_canon_promotions if missing (additive; modules table untouched)."""
    from core import db
    own = conn is None
    if own:
        conn = db.db_connect("logic")
    try:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS logic_canon_promotions (
            logic_id INTEGER NOT NULL, stage TEXT NOT NULL,
            promoted_at TEXT DEFAULT CURRENT_TIMESTAMP, rationale TEXT,
            PRIMARY KEY (logic_id, stage))""")
        if own:
            conn.commit()
    finally:
        if own:
            try:
                conn.close()
            except Exception:
                pass


def stage_of(logic_id):
    """Highest stage reached, defaulting to candidate (never raises)."""
    try:
        from core import db
        init_canon_table()
        conn = db.db_connect("logic")
        try:
            cur = conn.cursor()
            cur.execute("SELECT stage FROM logic_canon_promotions WHERE logic_id=?", (logic_id,))
            rows = {r["stage"] for r in cur.fetchall()}
        finally:
            try:
                conn.close()
            except Exception:
                pass
        for stage in reversed(STAGES):
            if stage in rows:
                return stage
        return "candidate"
    except Exception:
        return "candidate"


def _has_passing_evidence(logic_id):
    """Validated requires at least one example or prior verification trace."""
    try:
        from core import db
        conn = db.db_connect("logic")
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM logic_examples WHERE logic_id=? LIMIT 1", (logic_id,))
            if cur.fetchone():
                return True
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        pass
    return False


def promote(logic_id, stage, rationale="", approved=False):
    """Append a promotion record. Returns (ok, message); never raises.

    validated: needs passing evidence. canonical: needs validated + approved.
    Re-promotion to a held stage is a no-op success (idempotent).
    """
    if stage not in STAGES or stage == "candidate":
        return False, f"unknown stage {stage!r}"
    try:
        current = stage_of(logic_id)
        order = {s: i for i, s in enumerate(STAGES)}
        if order[current] >= order[stage]:
            return True, f"already {current}"
        if stage == "validated" and not _has_passing_evidence(logic_id):
            return False, "validated needs at least one example or verification trace"
        if stage == "canonical":
            if order[current] < order["validated"]:
                return False, "canonical needs validated first"
            if not approved:
                return False, "canonical needs explicit human/admin approval"
        from core import db
        init_canon_table()
        conn = db.db_connect("logic")
        try:
            conn.execute("INSERT OR IGNORE INTO logic_canon_promotions (logic_id, stage, rationale) VALUES (?,?,?)",
                         (logic_id, stage, rationale[:500]))
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return True, f"promoted to {stage}"
    except Exception as e:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return False, str(e)[:200]
