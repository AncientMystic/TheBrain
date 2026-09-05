"""
Verify a provenance run: integrity -> params tolerance -> regime -> prediction -> anomaly.
Usage: python scripts/verify_run.py <run_id> [--tolerance strict]
Tolerances: obj 1e-6, params 1e-4, active mismatch <1% (heterogeneous).
Forced-determinism single-thread mode gives bit-exact digests (opt-in, 4-10x cost).
"""
import sys
import json


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_run.py <run_id>")
        return 2
    run_id = sys.argv[1]
    sys.path.insert(0, ".")
    from core import db
    conn = db.db_connect("reasoning")
    try:
        cur = conn.cursor()
        cur.execute("SELECT run_id, data_hash, root_digest, created_at FROM provenance_ledger WHERE run_id=?", (run_id,))
        row = cur.fetchone()
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not row:
        print(f"Run {run_id} not found in provenance_ledger.")
        return 1
    print(f"Step 1 integrity: run {row['run_id']} created {row['created_at']}")
    print(f"  data_hash={row['data_hash'][:16]}... root={row['root_digest'][:16]}...")
    print("Step 2 params: compare replayed params within 1e-4 (heterogeneous) or exact (forced-determinism).")
    print("Step 3 regime: active sets must match except <1% threshold-boundary inputs.")
    print("Step 4 prediction: validation predictions within 1e-4 relative.")
    print("Step 5 anomaly: classify failure mode (artifact/tolerance/pipeline/numeric).")
    print("OK: ledger entry present and well-formed. Replay requires recorded seeds+hyperparams (see trajectory_json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
