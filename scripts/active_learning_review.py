
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import db
from extraction.gate import PrimeEvenGate
from core.spectral import compute_spectral_features
from core.embeddings import get_embeddings_batch

def review_uncertain_chunks(threshold_low=0.3, threshold_high=0.7, limit=50, batch_size=64):
    gate_path = Path(__file__).resolve().parent.parent / "models" / "gate.json"
    gate = PrimeEvenGate()
    if gate_path.exists():
        gate.load(gate_path)
    else:
        print("No trained gate found; cannot identify uncertain chunks.")
        return

    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, doc_hash, chunk_text FROM document_chunks")
    rows = cur.fetchall()
    conn.close()

    uncertain = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        texts = [r["chunk_text"] for r in batch]
        embs = get_embeddings_batch(texts, batch_size=batch_size)
        for r, emb in zip(batch, embs):
            if emb is not None:
                features = compute_spectral_features(np.array([emb], dtype=np.float32))
                w = gate.forward(features)
                if threshold_low < w < threshold_high:
                    uncertain.append((r["chunk_id"], r["doc_hash"], w, r["chunk_text"][:200]))

    uncertain.sort(key=lambda x: abs(x[2] - 0.5))
    print(f"Found {len(uncertain)} uncertain chunks. Showing first {min(limit, len(uncertain))}:")
    for chunk_id, doc_hash, w, preview in uncertain[:limit]:
        print(f"chunk {chunk_id} (doc {doc_hash}): weight={w:.3f} | {preview}")

if __name__ == "__main__":
    review_uncertain_chunks()
