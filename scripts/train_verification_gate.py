
import sys
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import db
from reasoning.verification_gate import VerificationGate, VERIFIER_NAMES

def collect_data():
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("SELECT features, labels FROM verification_gate_training_data")
    rows = cur.fetchall()
    conn.close()
    features = []
    labels_list = []
    for blob, labels_json in rows:
        feat = np.frombuffer(blob, dtype=np.float32)
        lab = json.loads(labels_json)
        features.append(feat)
        labels_list.append(lab)
    return features, labels_list

def train():
    features, labels_list = collect_data()
    if not features:
        print("No training data.")
        return

    X = np.array(features, dtype=np.float32)
    gate = VerificationGate()
    # For each verifier name that appears in training data, train a logistic regression.
    # For verifiers not seen, set their weights to produce ~1.0 output (no scaling).
    all_seen = set()
    for lab in labels_list:
        all_seen.update(lab.keys())

    for vname in all_seen:
        if vname not in VERIFIER_NAMES:
            print(f"Warning: verifier '{vname}' not recognized, skipping.")
            continue
        y = np.array([1 if lab.get(vname, 0) else 0 for lab in labels_list], dtype=np.float32)
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf.fit(X, y)
        # Set weights in gate
        idx = VERIFIER_NAMES.index(vname)
        gate.W[idx, 0] = clf.intercept_[0]
        gate.W[idx, 1:] = clf.coef_[0]

    # For verifiers not trained, set large positive intercept so output ~1.0
    for idx, vname in enumerate(VERIFIER_NAMES):
        if vname not in all_seen:
            gate.W[idx, 0] = 10.0
            gate.W[idx, 1:] = 0.0

    gate_path = Path(__file__).resolve().parent.parent / "models" / "verification_gate.json"
    gate_path.parent.mkdir(exist_ok=True)
    gate.save(gate_path)
    print(f"Verification gate saved to {gate_path}")

if __name__ == "__main__":
    train()
