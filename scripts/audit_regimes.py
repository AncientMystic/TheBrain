"""
Audit gate regime cube: marginal rates, joint co-occurrence, fingerprint, saturation.
Usage: python scripts/audit_regimes.py [--tau 0.5]
Reads gate_training_data features if present, else synthetic demo.
"""
import sys


def main():
    tau = 0.5
    if "--tau" in sys.argv:
        try:
            tau = float(sys.argv[sys.argv.index("--tau") + 1])
        except Exception:
            pass
    sys.path.insert(0, ".")
    import numpy as np
    from extraction.gate import PrimeEvenGate
    from core.regime_audit import activation_profile, marginal_rates, joint_matrix, loading_fingerprint, saturation_index
    # Load features
    try:
        from core import db
        conn = db.db_connect("key_facts")
        cur = conn.cursor()
        try:
            cur.execute("SELECT features FROM gate_training_data LIMIT 2000")
            rows = cur.fetchall()
        except Exception:
            rows = []
        conn.close()
        X = [np.frombuffer(r["features"], dtype=np.float32) for r in rows]
    except Exception:
        X = []
    import config
    gate = PrimeEvenGate()
    try:
        from pathlib import Path
        p = Path(config.BASE_DIR) / "models" / "gate.json"
        if p.exists():
            gate.load(str(p))
    except Exception:
        pass
    if not X:
        print("No gate_training_data found; using synthetic demo (8 corners).")
        rng = np.random.default_rng(0)
        X = [rng.standard_normal(44).astype(np.float32) for _ in range(64)]
    profiles = [activation_profile(gate, f) for f in X]
    mr = marginal_rates(profiles, tau=tau)
    jm = joint_matrix(profiles, tau=tau)
    fp = loading_fingerprint(gate)
    si = saturation_index(profiles)
    print(f"tau={tau} n={len(profiles)}")
    print(f"marginal top/gap/ups: {[round(v, 3) for v in mr]}")
    print("joint:")
    for row in jm:
        print("  " + " ".join(f"{v:.2f}" for v in row))
    print(f"fingerprint prime={fp['prime_support']:.3f} even={fp['even_support']:.3f} anchor={fp['anchor_coherence']:.3f}")
    print(f"saturation (>0.9/<0.1 all gates): {si:.3f} {'OK' if si > 0.85 else '(below 0.85 — see diagnostics)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
