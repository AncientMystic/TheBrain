"""Fill oct8 signatures for all embedded entities (phase 67/68).

Features per row: popularity, entity_type, depth, relation degree (both
directions), confidence heuristic (0.9 with description else 0.5),
provenance source count, alias count — blended 50/50 with the fixed
projection of the tangent vector (core.octonion). Resume-safe
(oct8 IS NULL), per-batch commits, fail-safe after 10 dead batches.

Usage: fill_oct8.py [--batch N] [--log PATH]
"""
import argparse
import sqlite3
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--log", default="")
    ap.add_argument("--db", default="mapping")
    args = ap.parse_args()
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None

    def say(msg):
        line = f"[fill_oct8] {msg}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    import numpy as _np
    from core import db as _db
    from core.octonion import signature, to_blob
    from core.embeddings import decode_embedding_blob
    conn = _db.db_connect(args.db)
    total = conn.execute(
        "SELECT COUNT(*) FROM entities WHERE emb IS NOT NULL").fetchone()[0]
    say(f"start: {total} embedded entities, batch={args.batch}")
    deg, src, als = {}, {}, {}
    try:
        for cid, n in conn.execute(
                "SELECT src_id, COUNT(*) FROM entity_relations GROUP BY src_id"):
            deg[cid] = deg.get(cid, 0) + n
        for cid, n in conn.execute(
                "SELECT dst_id, COUNT(*) FROM entity_relations GROUP BY dst_id"):
            deg[cid] = deg.get(cid, 0) + n
    except Exception as e:
        say(f"relations unavailable: {e}")
    try:
        for cid, n in conn.execute(
                "SELECT canonical_id, COUNT(DISTINCT source) FROM ingest_provenance GROUP BY canonical_id"):
            src[cid] = n
    except Exception as e:
        say(f"provenance unavailable: {e}")
    try:
        for cid, n in conn.execute(
                "SELECT canonical_id, COUNT(*) FROM aliases GROUP BY canonical_id"):
            als[cid] = n
    except Exception as e:
        say(f"aliases unavailable: {e}")
    done = done_base = conn.execute(
        "SELECT COUNT(*) FROM entities WHERE oct8 IS NOT NULL").fetchone()[0]
    fails = 0
    import time as _t
    t0 = _t.time()
    while True:
        rows = conn.execute(
            "SELECT canonical_id, entity_type, popularity, description,"
            " depth, emb FROM entities WHERE emb IS NOT NULL"
            " AND oct8 IS NULL LIMIT ?", (args.batch,)).fetchall()
        if not rows:
            break
        ok = 0
        for r in rows:
            cid = r[0]
            try:
                emb = decode_embedding_blob(r[5], context="fill_oct8")
                if emb is None:
                    continue
                o = signature(emb, popularity=r[2] or 0,
                              entity_type=r[1] or "",
                              depth=r[4] or 0.0, degree=deg.get(cid, 0),
                              confidence=0.9 if (r[3] or "") else 0.5,
                              n_sources=src.get(cid, 0),
                              n_aliases=als.get(cid, 0))
                conn.execute("UPDATE entities SET oct8=? WHERE canonical_id=?",
                             (to_blob(o), cid))
                ok += 1
            except Exception:
                continue
        conn.commit()
        done += len(rows)
        if ok == 0:
            fails += 1
            if fails >= 10:
                say("too many dead batches; stopping for resume later")
                conn.close()
                return 2
        else:
            fails = 0
        el = _t.time() - t0
        say(f"progress: {done}/{total} ({done * 100.0 / max(total, 1):.1f}%)"
            f" {done / max(el, 1e-3):.0f}/s")
    with_ = conn.execute(
        "SELECT COUNT(*) FROM entities WHERE oct8 IS NOT NULL").fetchone()[0]
    say(f"done: oct8 present on {with_}/{total}")
    conn.close()
    if logfh:
        logfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
