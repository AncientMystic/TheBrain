"""
Tune prime-even gate regularizer weights via k-fold CV on gate_training_data.

Grid: lam2==lam3 in {1e-2,1e-1,1,10}, lam1=0.1*lam2, lam4=lam2/8 (manuscript §7.3).
Reports prime-support, even-support, anchor coherence + validation loss.
Generic, no doc-specific hardcoding; writes best to models/gate.json.
"""
import itertools
import numpy as np
import config
from pathlib import Path


def _load_training():
    from core import db
    conn = db.db_connect("key_facts")
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT features, label FROM gate_training_data LIMIT 5000")
            rows = cur.fetchall()
        except Exception:
            rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass
    X, y = [], []
    for r in rows:
        try:
            f = np.frombuffer(r["features"], dtype=np.float32)
            X.append(f)
            y.append(int(r["label"]))
        except Exception:
            continue
    if not X:
        return None, None
    return np.stack(X), np.array(y, dtype=np.int64)


def _prime_stats(gate):
    import numpy as _np
    b = gate.beta[1:]
    g = gate.gamma[1:] if len(gate.gamma) > 1 else _np.zeros(1)
    prime = {2, 3, 5, 7, 11, 13, 17, 19}
    b_prime = sum(abs(b[i - 1]) for i in prime if i - 1 < len(b))
    b_total = float(_np.sum(_np.abs(b))) + 1e-8
    even = set(range(2, 22, 2))
    g_even = sum(abs(g[i - 1]) for i in even if i - 1 < len(g))
    g_total = float(_np.sum(_np.abs(g))) + 1e-8
    try:
        anchor = float(gate.delta[2])
        prime_vals = [float(gate.delta[i]) for i in prime if i < len(gate.delta)]
        avg = float(_np.mean(prime_vals)) if prime_vals else 0.0
        coh = 1.0 / (1.0 + abs(anchor - avg))
    except Exception:
        coh = 0.0
    return b_prime / b_total, g_even / g_total, coh


def main():
    from extraction.gate import PrimeEvenGate
    data = _load_training()
    if data[0] is None:
        print("No gate_training_data found. Run guided-learning first to collect labels.")
        return 1
    X, y = data
    print(f"Tuning on {len(y)} samples, {X.shape[1]} features")
    grid = [1e-2, 1e-1, 1.0, 10.0]
    best = None
    # 5-fold manual split (deterministic, no sklearn dep)
    idx = np.arange(len(y))
    folds = [idx[idx % 5 == k] for k in range(5)]
    for lam2 in grid:
        lam3 = lam2
        lam1 = 0.1 * lam2
        lam4 = lam2 / 8.0
        losses = []
        for k in range(5):
            va = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            gate = PrimeEvenGate()
            lr = float(getattr(config, "GATE_LR", 0.01))
            for _ in range(20):
                # Mini-batch full (small data) proximal steps
                gate.train_step(X[tr], y[tr], lr=lr, lam1=lam1, lam2=lam2, lam3=lam3, lam4=lam4)
            # Validation BCE
            outs = np.array([gate.forward(f) for f in X[va]], dtype=np.float32)
            eps = 1e-8
            loss = -np.mean(y[va] * np.log(outs + eps) + (1 - y[va]) * np.log(1 - outs + eps))
            losses.append(float(loss))
        mean_loss = float(np.mean(losses))
        # Train full for stats
        gate_full = PrimeEvenGate()
        for _ in range(30):
            gate_full.train_step(X, y, lr=float(getattr(config, "GATE_LR", 0.01)), lam1=lam1, lam2=lam2, lam3=lam3, lam4=lam4)
        ps, es, coh = _prime_stats(gate_full)
        print(f"lam2={lam2:g} lam1={lam1:g} lam4={lam4:g} val_loss={mean_loss:.4f} prime={ps:.2f} even={es:.2f} anchor_coh={coh:.2f}")
        if best is None or mean_loss < best[0]:
            best = (mean_loss, lam1, lam2, lam3, lam4, gate_full)
    mean_loss, lam1, lam2, lam3, lam4, gate_best = best
    out = Path(config.BASE_DIR) / "models" / "gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    gate_best.save(str(out))
    print(f"Best val_loss={mean_loss:.4f} lam1={lam1:g} lam2={lam2:g} lam3={lam3:g} lam4={lam4:g} -> {out}")
    ps, es, coh = _prime_stats(gate_best)
    print(f"Fingerprint: prime-support={ps:.3f} even-support={es:.3f} anchor-coherence={coh:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
