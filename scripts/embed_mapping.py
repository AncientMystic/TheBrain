"""Resumable embed pass for mapping entities (phase 62 follow-up).

Embeds every entities row with NULL emb using the configured mxbai
endpoint (hyperbolic space, dim 1024), stores float32 blobs in
entities.emb. Resume-safe: selects only emb IS NULL, commits per batch.
Stops (nonzero exit) after MAX_FAIL consecutive failed batches so a dead
backend never churns; rerun to resume.

Usage: embed_mapping.py [--limit N] [--batch N] [--log PATH]
Progress lines: done/total rate/s ETA, flushed to stdout and --log.
"""
import argparse
import sqlite3
import sys
import time

sys.path.insert(0, "A:/scripts/TheBrain")
import config
from core import db
from core.embeddings import get_embeddings_batch


def embed_text(eid, name, etype):
    base = (name or "").strip() or eid
    tag = (etype or "").strip()
    return f"{base} [{tag}]" if tag else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--log", default="")
    ap.add_argument("--db", default="mapping")
    args = ap.parse_args()
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None

    def say(msg):
        line = f"[embed_mapping] {msg}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    conn = db.db_connect(args.db)
    total = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    exp_dim = int(getattr(config, "EMBEDDING_DIM", 1024))
    say(f"start: {total} entities, batch={args.batch}, dim={exp_dim}"
        + (f", limit={args.limit}" if args.limit else ""))
    t0 = time.time()
    done_base = conn.execute(
        "SELECT COUNT(*) FROM entities WHERE emb IS NOT NULL").fetchone()[0]
    done = 0
    fails = 0
    MAX_FAIL = 10
    while True:
        rows = conn.execute(
            "SELECT canonical_id, display_name, entity_type FROM entities "
            "WHERE emb IS NULL LIMIT ?", (args.batch,)).fetchall()
        if not rows:
            break
        texts = [embed_text(r["canonical_id"], r["display_name"],
                            r["entity_type"]) for r in rows]
        try:
            embs = get_embeddings_batch(texts, space="hyperbolic",
                                        batch_size=args.batch)
        except Exception as e:
            say(f"batch error: {e}")
            embs = [None] * len(texts)
        ok = 0
        for r, emb in zip(rows, embs):
            if emb is not None and len(emb) == exp_dim:
                import numpy as _np
                conn.execute(
                    "UPDATE entities SET emb=? WHERE canonical_id=?",
                    (sqlite3.Binary(
                        _np.asarray(emb, dtype=_np.float32).tobytes()),
                     r["canonical_id"]))
                ok += 1
        conn.commit()
        done += len(rows)
        if args.limit and done >= args.limit:
            break
        if ok == 0:
            fails += 1
            say(f"batch all-failed ({fails}/{MAX_FAIL})")
            if fails >= MAX_FAIL:
                say("too many failed batches; stopping for resume later")
                conn.close()
                if logfh:
                    logfh.close()
                return 2
        else:
            fails = 0
        el = time.time() - t0
        rate = done / max(el, 1e-3)
        remain = total - done_base - done
        eta = remain / max(rate, 1e-3)
        say(f"progress: {done_base + done}/{total} "
            f"({(done_base + done) * 100.0 / max(total, 1):.1f}%) "
            f"{rate:.1f}/s eta={eta / 60:.0f}m")
    with_ = conn.execute(
        "SELECT COUNT(*) FROM entities WHERE emb IS NOT NULL").fetchone()[0]
    say(f"done: emb present on {with_}/{total}")
    conn.close()
    if logfh:
        logfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
