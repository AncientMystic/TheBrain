"""P5 proof gates runner (phase 74): PASS / FAIL / SKIP, never fake-pass.

Gates:
  ranking   - Curie + decoherence live proofs must exit 0 (real).
  latency   - p50 budgets on live DB: rank_candidates_scored < 1.0s,
              route_entities < 0.2s over 20 timed calls each (real).
  hallucination / faithfulness - SKIP: need an LLM backend plus labeled
              ground-truth fixtures, neither present in this harness.
              Recorded as SKIP with reason, not pass.

Exit 0 when nothing FAILs (SKIPs allowed); exit 1 on any FAIL.
Usage: proof_gates.py
"""
import sys
import time

sys.path.insert(0, "A:/scripts/TheBrain")


def _time(fn, n=20):
    ds = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ds.append(time.perf_counter() - t0)
    ds.sort()
    return ds[len(ds) // 2], max(ds)


def gate_ranking():
    try:
        sys.path.insert(0, "A:/scripts/TheBrain/scripts")
        import prove_curie_paris as P
        import prove_decoherence_live as D
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            assert P.main() == 0
            assert D.main() == 0
        return "PASS", "curie + decoherence proofs exit 0"
    except AssertionError as e:
        return "FAIL", f"proof assertion: {str(e)[:160]}"
    except SystemExit as e:
        return ("PASS", "proofs exit 0") if e.code == 0 else ("FAIL", f"exit {e.code}")
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:160]}"


def gate_latency():
    try:
        from core import db
        from core.mapping_lookup import rank_candidates_scored
        from core.octonion import from_blob
        from core.e8 import route_entities
        conn = db.db_connect("mapping")
        try:
            mentions = ["Paris", "Curie", "Marie Curie", "Texas", "London",
                        "Einstein", "Berlin", "Shakespeare", "Tokyo", "Mozart"]
            p50_rank, pmax_rank = _time(
                lambda: rank_candidates_scored(
                    mentions[int(time.perf_counter() * 997) % len(mentions)],
                    limit=8, conn=conn))
            rows = conn.execute("SELECT oct8 FROM entities WHERE oct8 IS NOT"
                                " NULL LIMIT 20").fetchall()
            vecs = [from_blob(bytes(r[0])) for r in rows]
            p50_route, pmax_route = _time(
                lambda: route_entities(
                    vecs[int(time.perf_counter() * 613) % len(vecs)],
                    conn, top_cells=2, limit=50))
            detail = (f"rank p50={p50_rank * 1000:.0f}ms max={pmax_rank * 1000:.0f}ms;"
                      f" route p50={p50_route * 1000:.0f}ms max={pmax_route * 1000:.0f}ms")
            if p50_rank < 1.0 and p50_route < 0.2:
                return "PASS", detail
            return "FAIL", detail + " (budget exceeded)"
        finally:
            conn.close()
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {str(e)[:160]}"


GATES = [
    ("ranking", "Curie + decoherence live proofs", gate_ranking),
    ("latency", "p50 budgets on live DB", gate_latency),
    ("hallucination", "needs LLM + labeled fixtures", lambda: ("SKIP", "no ground-truth fixture set")),
    ("faithfulness", "needs LLM + citation fixtures", lambda: ("SKIP", "no citation fixture set")),
]


def main():
    results = [(name, desc, fn()) for name, desc, fn in GATES]
    for name, desc, (status, detail) in results:
        print(f"[{status}] {name:14s} {desc} -- {detail}", flush=True)
    failed = [n for n, _, (s, _) in results if s == "FAIL"]
    print(f"proof_gates: {'FAIL: ' + ','.join(failed) if failed else 'no failures'}"
          f" ({sum(1 for _, _, (s, _) in results if s == 'SKIP')} skipped)",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
