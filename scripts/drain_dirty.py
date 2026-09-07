"""Dirty-queue drain (phase 73): Enneagram resolver as consumer.

Default is dry-run (zero writes): classifies every pending dirty_mentions
row and prints the recommendation. --apply writes Law-of-Three records
via enneagram.resolve_claim, but ONLY for the safe class (exactly one
candidate reached by exact alias match); everything else stays queued
for human review. Never deletes rows.

Usage: drain_dirty.py [--apply] [--limit N] [db-path]
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def classify(conn, surface):
    """(recommendation, canonical_or_reason). Read-only."""
    from core.mapping_lookup import mapping_candidates_fn, normalize_mention
    try:
        cands = mapping_candidates_fn(surface, limit=5, conn=conn) or []
    except Exception:
        return "review", "candidate lookup failed"
    if not cands:
        return "defer", "no candidates"
    if len(cands) == 1:
        try:
            norm = normalize_mention(surface)
            row = conn.execute("SELECT 1 FROM aliases WHERE alias_norm=?"
                               " AND canonical_id=?", (norm, cands[0][0])
                               ).fetchone()
            if row:
                return "accept", cands[0][0]
        except Exception:
            pass
    return "review", f"{len(cands)} candidates"


def drain(conn, apply=False, limit=0):
    from core import enneagram as G
    rows = conn.execute("SELECT surface, doc_id, proposed_canonical"
                        " FROM dirty_mentions WHERE status LIKE 'pending%'"
                        " OR status LIKE 'open-%'"
                        + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    counts = {"accept": 0, "defer": 0, "review": 0, "applied": 0}
    for surface, doc_id, proposed in rows:
        rec, info = classify(conn, surface)
        counts[rec] += 1
        print(f"[{rec}] {surface[:40]:40s} {doc_id[:16]:16s} {str(info)[:40]}",
              flush=True)
        if apply and rec == "accept":
            if G.resolve_claim(conn, surface, doc_id, "accept",
                               f"drain:auto exact -> {info}"):
                counts["applied"] += 1
    print(f"drain: {len(rows)} rows (accept={counts['accept']}"
          f" defer={counts['defer']} review={counts['review']}"
          f" applied={counts['applied']})"
          f"{'' if apply else ' DRY-RUN, zero writes'}", flush=True)
    return counts


def main(argv):
    import sqlite3
    apply = "--apply" in argv
    rest = [a for a in argv if a not in ("--apply",) and not a.startswith("--limit")]
    limit = 0
    for a in argv:
        if a.startswith("--limit"):
            limit = int(a.split("=", 1)[1] if "=" in a else argv[argv.index(a) + 1])
    if rest:
        conn = sqlite3.connect(rest[0])
        conn.row_factory = None
    else:
        from core import db as _db
        conn = _db.db_connect("mapping")
    try:
        counts = drain(conn, apply=apply, limit=limit)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
