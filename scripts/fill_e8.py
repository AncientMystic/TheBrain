"""Fill e8_key routing cells from oct8 signatures (phase 68).

Resume-safe (e8_key IS NULL), per-batch commits, fail-safe after 10 dead
batches. Pure numpy; needs oct8 only.

Usage: fill_e8.py [--batch N] [--log PATH]
"""
import argparse
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4000)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None

    def say(msg):
        line = f"[fill_e8] {msg}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    from core import db as _db
    from core.octonion import from_blob
    from core.e8 import quantize
    import time as _t
    conn = _db.db_connect("mapping")
    total = conn.execute(
        "SELECT COUNT(*) FROM entities WHERE oct8 IS NOT NULL").fetchone()[0]
    say(f"start: {total} signed entities, batch={args.batch}")
    done = conn.execute(
        "SELECT COUNT(*) FROM entities WHERE e8_key IS NOT NULL").fetchone()[0]
    fails = 0
    t0 = _t.time()
    while True:
        rows = conn.execute(
            "SELECT canonical_id, oct8 FROM entities WHERE oct8 IS NOT NULL"
            " AND e8_key IS NULL LIMIT ?", (args.batch,)).fetchall()
        if not rows:
            break
        ok = 0
        for cid, blob in rows:
            try:
                q = quantize(from_blob(bytes(blob)))
                if q is None:
                    continue
                conn.execute("UPDATE entities SET e8_key=? WHERE canonical_id=?",
                             (q[0], cid))
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
        "SELECT COUNT(*) FROM entities WHERE e8_key IS NOT NULL").fetchone()[0]
    cells = conn.execute(
        "SELECT COUNT(DISTINCT e8_key) FROM entities").fetchone()[0]
    say(f"done: e8_key on {with_}/{total} across {cells} cells")
    conn.close()
    if logfh:
        logfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
