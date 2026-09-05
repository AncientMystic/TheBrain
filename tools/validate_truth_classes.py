"""
CI validator for truth-class integrity (presence + origin invariants, not correctness).

Scans samples of key_facts and verification_standards, derives truth classes,
and fails on unknown classes or admin/verified origin violations. Correctness
of individual facts remains the verifier's job — this gate catches structural
drift (bad enum values, collapsed axes). Run: python tools/validate_truth_classes.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE = 2000


def _scan(conn, table, cols):
    from core.truth import validate_fact
    cur = conn.cursor()
    try:
        cur.execute(f"PRAGMA table_info({table})")
        have = {r[1] for r in cur.fetchall()}
    except Exception:
        have = set(cols)
    use = [c for c in cols if c in have] or cols[:1]
    try:
        cur.execute(f"SELECT {', '.join(use)} FROM {table} LIMIT {SAMPLE}")
        rows = cur.fetchall()
    except Exception as e:
        return [f"{table}: unreadable ({e})"]
    problems = []
    for i, r in enumerate(rows):
        try:
            fact = {c: r[c] for c in cols if c in r.keys()}
        except Exception:
            fact = dict(r)
        for p in validate_fact(fact):
            problems.append(f"{table} row {i}: {p}")
            if len(problems) > 20:
                problems.append("... (truncated)")
                return problems
    return problems


def main():
    from core import db
    problems = []
    conn = db.db_connect("key_facts")
    try:
        problems += _scan(conn, "key_facts", ["fact_text", "verification_status", "truth_status", "verified_by", "source_span"])
    finally:
        try:
            conn.close()
        except Exception:
            pass
    conn = db.db_connect("verification_standards")
    try:
        problems += _scan(conn, "verified_standards", ["statement", "truth_status", "source_type"])
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if problems:
        print("TRUTH-CLASS VIOLATIONS:")
        for p in problems:
            print(" -", p)
        return 1
    print(f"truth classes OK (sampled up to {SAMPLE} rows per table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
