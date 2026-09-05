"""
Append-only fact valuations with supersedes links and proof artifacts.

Current tables stay the read-compatible present state (no migration, no drops);
every material verification outcome also appends a version row carrying its
proof artifact (layers, confidence, truth class, timestamp). History is never
rewritten: corrections supersede, they do not overwrite. Generic, bounded.
"""
import hashlib
import json
import time
import logging

logger = logging.getLogger(__name__)


def init_versions_table(conn=None):
    """Create fact_versions if missing (additive; existing tables untouched)."""
    from core import db
    own = conn is None
    if own:
        conn = db.db_connect("key_facts")
    try:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS fact_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_id INTEGER, fact_text TEXT NOT NULL, canonical_value TEXT,
            source_span TEXT, confidence REAL DEFAULT 0.0, truth_class TEXT,
            verification_status TEXT, verification_artifact_json TEXT,
            supersedes_version_id INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_versions_fact ON fact_versions(fact_id, version_id)")
        if own:
            conn.commit()
    finally:
        if own:
            try:
                conn.close()
            except Exception:
                pass


def _material_hash(fact):
    h = hashlib.sha256()
    for k in ("fact_text", "canonical_value", "source_span", "confidence",
              "truth_class", "verification_status"):
        h.update(str(fact.get(k, "")).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def _proof_artifact(fact):
    try:
        return json.dumps({
            "layers": fact.get("verification_layers", []),
            "confidence_final": fact.get("confidence_final", fact.get("confidence", 0)),
            "verified_by": fact.get("verified_by", ""),
            "truth_class": fact.get("truth_class", ""),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, default=str)[:20000]
    except Exception:
        return "{}"


def record_version(fact, fact_id=None):
    """Append a version row when material content differs from latest.

    Returns version_id, existing id when unchanged, or None when disabled/failed.
    Never raises; never mutates existing rows.
    """
    try:
        import config as _cfg
        if not getattr(_cfg, "FACT_VERSIONING_ENABLED", True):
            return None
    except Exception:
        pass
    try:
        from core import db
        init_versions_table()
        fid = fact_id if fact_id is not None else fact.get("fact_id")
        digest = _material_hash(fact)
        conn = db.db_connect("key_facts")
        try:
            cur = conn.cursor()
            latest = None
            if fid is not None:
                cur.execute("""SELECT version_id, fact_text, canonical_value, source_span,
                                      confidence, truth_class, verification_status
                               FROM fact_versions WHERE fact_id=? ORDER BY version_id DESC LIMIT 1""",
                            (fid,))
                latest = cur.fetchone()
            if latest:
                prev = {"fact_text": latest["fact_text"], "canonical_value": latest["canonical_value"],
                        "source_span": latest["source_span"], "confidence": latest["confidence"],
                        "truth_class": latest["truth_class"], "verification_status": latest["verification_status"]}
                if _material_hash(prev) == digest:
                    return int(latest["version_id"])
                supersedes = int(latest["version_id"])
            else:
                supersedes = None
            cur.execute("""INSERT INTO fact_versions
                (fact_id, fact_text, canonical_value, source_span, confidence, truth_class,
                 verification_status, verification_artifact_json, supersedes_version_id)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (fid, str(fact.get("fact_text", "")), str(fact.get("canonical_value", "")),
                 str(fact.get("source_span", "")), float(fact.get("confidence", 0) or 0),
                 str(fact.get("truth_class", "")), str(fact.get("verification_status", "")),
                 _proof_artifact(fact), supersedes))
            vid = int(cur.lastrowid)
            conn.commit()
            return vid
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Version record skipped: {e}", exc_info=True)
        return None


def get_history(fact_id, limit=50):
    """Newest-first version rows for a fact (empty list when none)."""
    try:
        from core import db
        conn = db.db_connect("key_facts")
        try:
            cur = conn.cursor()
            cur.execute("""SELECT version_id, fact_text, verification_status, truth_class,
                                  confidence, supersedes_version_id, created_at
                           FROM fact_versions WHERE fact_id=? ORDER BY version_id DESC LIMIT ?""",
                        (fact_id, max(1, min(int(limit), 200))))
            return [dict(r) for r in cur.fetchall()]
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        return []
