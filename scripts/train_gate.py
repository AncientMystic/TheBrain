
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import config
from extraction.gate import PrimeEvenGate
from core import db

def collect_training_data():
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("SELECT features, label FROM gate_training_data")
    rows = cur.fetchall()
    conn.close()
    features = []
    labels = []
    for blob, label in rows:
        feat = np.frombuffer(blob, dtype=np.float32)
        features.append(feat)
        labels.append(label)
    if not features:
        return np.array([], dtype=np.float32).reshape(0,44), np.array([], dtype=np.float32)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)

def train():
    X, y = collect_training_data()
    if len(X) == 0:
        print("No training data collected yet.")
        return
    gate = PrimeEvenGate()
    indices = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Training gate on {len(X_train)} samples, validating on {len(X_val)}")
    best_loss = float('inf')
    patience = 10
    no_improve = 0
    for epoch in range(200):
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]
        batch_size = int(getattr(config, "GATE_BATCH_SIZE", 32))
        _lr = float(getattr(config, "GATE_LR", 0.01))
        _lam1 = float(getattr(config, "GATE_LAM1", 0.01))
        _lam2 = float(getattr(config, "GATE_LAM2", 0.1))
        _lam3 = float(getattr(config, "GATE_LAM3", 0.1))
        _lam4 = float(getattr(config, "GATE_LAM4", 0.05))
        total_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]
            loss = gate.train_step(xb, yb, lr=_lr, lam1=_lam1, lam2=_lam2, lam3=_lam3, lam4=_lam4)
            total_loss += loss * len(xb)
        avg_loss = total_loss / len(X_train)

        val_outputs = np.array([gate.forward(f) for f in X_val])
        val_loss = -np.mean(y_val * np.log(val_outputs + 1e-8) + (1-y_val) * np.log(1-val_outputs + 1e-8))
        if epoch % 10 == 0:
            try:
                from core.regime_audit import loading_fingerprint
                fp = loading_fingerprint(gate)
                print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"prime={fp['prime_support']:.2f} even={fp['even_support']:.2f} anchor={fp['anchor_coherence']:.2f}")
            except Exception:
                print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping")
                break

    gate_path = Path(config.BASE_DIR) / "models" / "gate.json"
    gate_path.parent.mkdir(exist_ok=True)
    gate.save(gate_path)
    print(f"Gate saved to {gate_path}")

if __name__ == "__main__":
    train()
