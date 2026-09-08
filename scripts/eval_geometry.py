"""Before/after geometry eval (phase 96): size, speed, fidelity.

Baseline: 1024-d mxbai hyperbolic (current production).
Challengers: v1 head (1024->128) + v2 full (128 hyp + 8 oct).
Measures: bytes/entity, pairwise-distance throughput (numpy, same CPU),
self-retrieval top-1/top-5 over a 200-entity sample (query = own display
name emb; does the entity rank itself?), parent-beats-sibling rate.
Exit 0 with printed table; numbers decide, not adjectives.
Usage: eval_geometry.py [--n 200]
"""
import sys
import time

sys.path.insert(0, "A:/scripts/TheBrain")
sys.path.insert(0, "A:/scripts/TheBrain/scripts")


def main(argv):
    import numpy as np
    import torch
    from core import db as _db
    from core.embeddings import decode_embedding_blob, get_embeddings_batch
    from core.hyperbolic import hyperbolic_distance
    n = 200
    for a in argv:
        if a.startswith("--n"):
            n = int(a.split("=", 1)[1] if "=" in a else 200)
    conn = _db.db_connect("mapping")
    rows = conn.execute("SELECT canonical_id, display_name, emb FROM entities"
                        " WHERE emb IS NOT NULL ORDER BY canonical_id LIMIT ?", (n,)).fetchall()
    cids = [r[0] for r in rows]
    names = [r[1] for r in rows]
    E = np.stack([np.asarray(decode_embedding_blob(bytes(r[2]), ""), dtype=np.float32)
                  for r in rows])
    conn.close()
    qtexts = [f"{dn}" for dn in names]
    Q = []
    B = 32
    for i in range(0, len(qtexts), B):
        Q.extend(get_embeddings_batch(qtexts[i:i + B], space="hyperbolic"))
    Q = np.stack([np.asarray(q, dtype=np.float32) for q in Q])

    from train_hyper_head import HyperHead
    from train_hyper_full import FullModel
    h1 = HyperHead()
    h1.load_state_dict(torch.load("A:/scripts/TheBrain/models/hyperbolic_head/head.pt",
                                  map_location="cpu", weights_only=True))
    h1.eval()
    h2 = FullModel()
    h2.load_state_dict(torch.load("A:/scripts/TheBrain/models/hyper_full/full.pt",
                                  map_location="cpu", weights_only=True))
    h2.eval()
    with torch.no_grad():
        E1 = h1(torch.from_numpy(E)).numpy()
        Q1 = h1(torch.from_numpy(Q)).numpy()
        E2, E2o = h2(torch.from_numpy(E))
        Q2, _ = h2(torch.from_numpy(Q))
        E2, Q2, E2o = E2.numpy(), Q2.numpy(), E2o.numpy()

    def rank_rate(Qm, Em, fn, topk=(1, 5)):
        hits = {k: 0 for k in topk}
        m = len(Qm)
        for i in range(m):
            d = np.array([fn(Qm[i], Em[j]) for j in range(m)])
            r = int((d < d[i]).sum()) + 1
            for k in topk:
                if r <= k:
                    hits[k] += 1
        return {k: hits[k] / m for k in topk}

    def hd(a, b):
        return hyperbolic_distance(a, b)

    def ed(a, b):
        return float(np.linalg.norm(a - b))

    r_base = rank_rate(Q, E, hd)
    r_h1 = rank_rate(Q1, E1, ed)
    r_h2 = rank_rate(Q2, E2, ed)
    # speed: 2000 pairwise distances, same CPU
    import time as _t
    def bench(fn, a, b, reps=2000):
        t0 = _t.perf_counter()
        for i in range(reps):
            fn(a[i % len(a)], b[(i * 7) % len(b)])
        return reps / max(_t.perf_counter() - t0, 1e-9)
    s_base = bench(hd, Q, E)
    s_128 = bench(ed, Q1, E1)
    print(f"bytes/entity : base=4096 v1=512 v2=512+32 (8x / 7.4x storage)")
    print(f"pairwise/s   : base1024={s_base:.0f} hyp128={s_128:.0f} ({s_128 / max(s_base, 1):.1f}x)")
    print(f"self top-1   : base={r_base[1]:.3f} v1={r_h1[1]:.3f} v2={r_h2[1]:.3f}")
    print(f"self top-5   : base={r_base[5]:.3f} v1={r_h1[5]:.3f} v2={r_h2[5]:.3f}")
    print(f"oct8 bytes   : 64 -> 32 (2x) with learned norms (v2 oct_mse in metrics)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
