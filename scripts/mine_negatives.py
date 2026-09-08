"""Sibling-negative miner (phase 95): hard negatives for entailment training.

A hard negative shares the anchor's hypernym parent but is a different
entity (Paris TX vs Paris France under 'city'; Curie vs Einstein under
'physicist'). Random negatives are too easy; siblings force the boundary.
Caps prevent combinatorial explosion (coarse parents like 'person' have
tens of thousands of children): per parent max 200 children (stable
ORDER BY), per anchor max 3 negatives, round-robin. Idempotent table
sibling_negatives + JSONL export. Never deletes.

Usage: mine_negatives.py [--per-parent N] [--per-anchor N] [--export PATH] [--log PATH]
"""
import argparse
import json
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

PER_PARENT = 200
PER_ANCHOR = 3


def ensure_table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS sibling_negatives("
                 " anchor_cid TEXT NOT NULL, positive_label TEXT NOT NULL,"
                 " negative_cid TEXT NOT NULL,"
                 " PRIMARY KEY (anchor_cid, positive_label, negative_cid))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sibneg_anchor ON sibling_negatives(anchor_cid)")
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ent_display ON entities(display_name)")
    except Exception:
        pass
    conn.commit()


def mine(conn, per_parent=PER_PARENT, per_anchor=PER_ANCHOR, log=None):
    def say(m):
        line = f"[mine_neg] {m}"
        print(line, flush=True)
        if log:
            log.write(line + "\n")
            log.flush()

    ensure_table(conn)
    total = mine_shared_parent(conn, per_parent, per_anchor, say)
    total += mine_confusables(conn, say)
    say(f"sibling negatives written (this run): {total}")
    tot = conn.execute("SELECT COUNT(*) FROM sibling_negatives").fetchone()[0]
    say(f"sibling_negatives total: {tot}")
    return total


def mine_shared_parent(conn, per_parent, per_anchor, say):
    parents = [r[0] for r in conn.execute(
        "SELECT parent_label FROM hypernym_edges GROUP BY parent_label"
        " HAVING COUNT(DISTINCT child_cid) >= 2")]
    say(f"parents with 2+ children: {len(parents)}")
    total = 0
    for p in parents:
        kids = [r[0] for r in conn.execute(
            "SELECT DISTINCT child_cid FROM hypernym_edges WHERE parent_label=?"
            " ORDER BY child_cid LIMIT ?", (p, per_parent))]
        if len(kids) < 2:
            continue
        n = len(kids)
        for i, a in enumerate(kids):
            made = 0
            for j in range(1, n):
                b = kids[(i + j) % n]
                if b == a:
                    continue
                try:
                    conn.execute("INSERT OR IGNORE INTO sibling_negatives"
                                 " (anchor_cid, positive_label, negative_cid)"
                                 " VALUES (?,?,?)", (a, p, b))
                    made += 1
                    total += 1
                except Exception:
                    continue
                if made >= per_anchor:
                    break
        conn.commit()
    return total


def mine_confusables(conn, say):
    """Same normalized display name, different canonical = true confusables
    (Paris FR/TX, Curie person/place). Highest-value negatives; exhaustive
    within each name group (groups are small). Positive = shared name."""
    total = 0
    groups = conn.execute("SELECT display_name, COUNT(DISTINCT canonical_id) AS n"
                          " FROM entities GROUP BY display_name"
                          " HAVING n > 1 ORDER BY n DESC").fetchall()
    say(f"confusable name groups: {len(groups)}")
    for name, n in groups:
        kids = [r[0] for r in conn.execute(
            "SELECT DISTINCT canonical_id FROM entities WHERE display_name=?"
            " ORDER BY canonical_id LIMIT 60", (name,))]
        for i, a in enumerate(kids):
            for b in kids:
                if b == a:
                    continue
                try:
                    conn.execute("INSERT OR IGNORE INTO sibling_negatives"
                                 " (anchor_cid, positive_label, negative_cid)"
                                 " VALUES (?,?,?)", (a, f"name:{name}", b))
                    total += 1
                except Exception:
                    continue
        conn.commit()
    say(f"confusable negatives: {total}")
    return total


def export_jsonl(conn, path):
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in conn.execute("SELECT anchor_cid, positive_label, negative_cid"
                              " FROM sibling_negatives"):
            fh.write(json.dumps({"anchor": r[0], "positive": r[1],
                                 "negative": r[2]}, ensure_ascii=True) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-parent", type=int, default=PER_PARENT)
    ap.add_argument("--per-anchor", type=int, default=PER_ANCHOR)
    ap.add_argument("--export", default="")
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None
    from core import db as _db
    _conn = _db.db_connect("mapping")
    mine(_conn, per_parent=args.per_parent, per_anchor=args.per_anchor, log=logfh)
    if args.export:
        _n = export_jsonl(_conn, args.export)
        print(f"[mine_neg] exported {_n} -> {args.export}", flush=True)
    _conn.close()
    if logfh:
        logfh.close()
