"""Full hyperbolic+octonion model (phase 96): joint multi-task geometry.

Shared trunk 1024->512->256 with two paired heads:
  H: 256->128 + expmap, entailment-cone + contrastive loss (hierarchy).
  O: 256->8 octonion branch, supervised by stored oct8 signatures
     (MSE + norm correlation) — the 8th-dimension mapping.
  Pairing loss: Pearson correlation between batch pairwise
  mobius-distances (H) and euclidean distances (O) — the two geometries
  must agree on neighborhood structure (correlation pairing).
Reuses v1 mobius ops + parent-emb cache. Saves models/hyper_full/.
Usage: train_hyper_full.py [--epochs N] [--batch N] [--log PATH]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "A:/scripts/TheBrain")
sys.path.insert(0, "A:/scripts/TheBrain/scripts")

from train_hyper_head import (mobius_dist, expmap0, cone_aperture, angle_at,
                              OUT_DIR as V1_DIR)

DIM_IN, DIM_H, MARGIN = 1024, 128, 1.0
OUT_DIR = "A:/scripts/TheBrain/models/hyper_full"


class Trunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, 512), torch.nn.LayerNorm(512), torch.nn.GELU(),
            torch.nn.Linear(512, 512), torch.nn.LayerNorm(512), torch.nn.GELU(),
            torch.nn.Linear(512, 256), torch.nn.LayerNorm(256), torch.nn.GELU())

    def forward(self, x):
        return self.net(x)


class FullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.hyp = torch.nn.Linear(256, DIM_H)
        self.oct = torch.nn.Linear(256, 8)
        self.log_scale = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        t = self.trunk(x)
        return expmap0(self.hyp(t) * self.log_scale.exp()), self.oct(t)


def pcorr(a, b, eps=1e-4):
    a = a - a.mean()
    b = b - b.mean()
    den = a.std() * b.std()
    if float(den.detach()) < eps:
        return torch.zeros((), device=a.device)
    return (a * b).mean() / (den + eps)


def batch_pdist(h, o):
    """Pairwise mobius (H) vs SQUARED-euclidean (O) distances, strict upper triangle.

    Vectorized over index pairs: the diagonal (self-pairs, exactly-zero
    norm -> NaN gradients via SqrtBackward) is never constructed.
    Squared-euclidean (no sqrt) keeps the O-side smooth everywhere:
    identical rows give distance 0 with zero gradient, never inf.
    Monotonicity is all the correlation loss needs.
    """
    import torch as _t
    n = h.shape[0]
    iu = _t.triu_indices(n, n, offset=1, device=h.device)
    dh = mobius_dist(h[iu[0]], h[iu[1]])
    de = ((o[iu[0]] - o[iu[1]]) ** 2).sum(-1)
    return dh, de


def load_quads(parent_embs):
    """[(anchor_emb, parent_emb, neg_emb, oct8_target)] — oct8 from mapping."""
    import numpy as np
    from core import db as _db
    from core.embeddings import decode_embedding_blob
    from core.octonion import from_blob
    conn = _db.db_connect("mapping")
    out = []
    with open("A:/scripts/TheBrain/data/sibling_neg.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                a = conn.execute("SELECT emb, oct8 FROM entities WHERE canonical_id=?",
                                 (d["anchor"],)).fetchone()
                b = conn.execute("SELECT emb FROM entities WHERE canonical_id=?",
                                 (d["negative"],)).fetchone()
                pe = parent_embs.get(d["positive"].strip())
                if not (a and a[0] and a[1] and b and b[0] and pe is not None):
                    continue
                ae = decode_embedding_blob(a[0], "")
                be = decode_embedding_blob(b[0], "")
                oe = from_blob(bytes(a[1]))
                if ae is None or be is None:
                    continue
                out.append((ae.astype("float32"), pe.astype("float32"),
                            be.astype("float32"), oe.astype("float32")))
            except Exception:
                continue
    conn.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None

    def say(m):
        line = f"[hyperfull] {m}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"device={dev}")
    import numpy as np
    _pl = os.path.join(V1_DIR, "parent_labels.json")
    _pz = os.path.join(V1_DIR, "parent_embs.npz")
    if not (os.path.exists(_pl) and os.path.exists(_pz)):
        say("parent cache missing (v1 still building); aborting, retry after v1")
        return 3
    _z = np.load(_pz)
    _labels = json.load(open(_pl))
    parent_embs = {l: _z[f"p{i}"] for i, l in enumerate(_labels)}
    say(f"parent vectors (shared v1 cache): {len(parent_embs)}")
    quads = load_quads(parent_embs)
    say(f"quads: {len(quads)}")
    assert len(quads) > 1000, "too few quads"
    A = np.stack([t[0] for t in quads])
    P = np.stack([t[1] for t in quads])
    N = np.stack([t[2] for t in quads])
    O = np.stack([t[3] for t in quads])
    On = (O - O.mean(0)) / (O.std(0) + 1e-6)
    n = len(A)
    idx = np.random.default_rng(11).permutation(n)
    cut = int(n * 0.9)
    tr, te = idx[:cut], idx[cut:]
    model = FullModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    B = args.batch
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(cut)
        tot, nb = 0.0, 0
        for i in range(0, cut, B):
            bi = perm[i:i + B]
            a = torch.from_numpy(A[tr[bi.numpy()]]).to(dev)
            p = torch.from_numpy(P[tr[bi.numpy()]]).to(dev)
            ne = torch.from_numpy(N[tr[bi.numpy()]]).to(dev)
            ot = torch.from_numpy(On[tr[bi.numpy()]]).to(dev)
            ha, hp, hn = (model(a)[0], model(p)[0], model(ne)[0])
            _, oa = model(a)[1], model(a)[1]
            d_pos = mobius_dist(ha, hp)
            d_neg = mobius_dist(ha, hn)
            loss_con = torch.relu(d_pos - d_neg + MARGIN).mean()
            loss_ent = torch.relu(angle_at(hp, ha) - cone_aperture(hp)).mean()
            loss_oct = ((oa - ot) ** 2).mean()
            loss_norm = ((oa.norm(dim=-1) - torch.from_numpy(
                np.linalg.norm(O[tr[bi.numpy()]], axis=1)).float().to(dev)
                .div(10.0)) ** 2).mean()
            # Subsample 32 rows: O(n^2) pairing is the bottleneck.
            _k = min(32, ha.shape[0])
            dhg, _ = batch_pdist(ha[:_k], oa.detach()[:_k])
            _, dev_s = batch_pdist(ha.detach()[:_k], oa[:_k])
            loss_corr = 1 - pcorr(dhg, dev_s)
            loss = loss_con + 0.5 * loss_ent + 0.3 * loss_oct + 0.1 * loss_norm + 0.2 * loss_corr
            opt.zero_grad()
            if not torch.isfinite(loss):
                say("non-finite loss, skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(bi)
            nb += len(bi)
        say(f"epoch {ep + 1}/{args.epochs} loss={tot / max(nb, 1):.4f}")
    model.eval()
    with torch.no_grad():
        correct = ent_ok = 0
        oct_err, corrs, m = 0.0, [], len(te)
        for i in range(0, m, B):
            bi = te[i:i + B]
            a = torch.from_numpy(A[bi]).to(dev)
            p = torch.from_numpy(P[bi]).to(dev)
            ne = torch.from_numpy(N[bi]).to(dev)
            ha, hp, hn = model(a)[0], model(p)[0], model(ne)[0]
            _, oa = model(a)[1], model(a)[1]
            correct += int((mobius_dist(ha, hp) < mobius_dist(ha, hn)).sum())
            ent_ok += int((angle_at(hp, ha) <= cone_aperture(hp)).sum())
            ot = torch.from_numpy(On[bi]).to(dev)
            oct_err += float(((oa - ot) ** 2).mean()) * len(bi)
            # chunked pairing: full-batch triu on thousands of rows OOMs
            _ch = 512
            _cs = []
            for _s in range(0, len(bi), _ch):
                _sl = slice(_s, _s + _ch)
                _dh, _de = batch_pdist(ha[_sl], oa[_sl])
                _cs.append(float(pcorr(_dh, _de)))
            import numpy as _np2
            corrs.append(float(_np2.mean(_cs)) if _cs else 0.0)
        import numpy as _np
        metrics = {"eval_n": m, "retrieval_acc": correct / max(m, 1),
                   "entailment_rate": ent_ok / max(m, 1),
                   "oct_mse": oct_err / max(m, 1),
                   "cross_corr": float(_np.mean(corrs)) if corrs else 0.0}
    say(f"EVAL {metrics}")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "full.pt"))
    json.dump(metrics, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=1)
    if logfh:
        logfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
