"""Factorized hyperbolic+identity model (phase 97): Schmidt decomposition.

Shared trunk 1024->512->256, then two disentangled 64-d subspaces:
  HIER (hyperbolic): entailment-cone + sibling-contrastive loss.
     The environment: shared hierarchy for routing.
  IDENT (euclidean, unit norm): InfoNCE instance discrimination.
     Positives = two dropout-augmented views of the same anchor
     (SimCLR-style, no extra embeddings needed); in-batch siblings
     as hardest negatives. The traced-over remainder: identity with
     shared context marginalized out.
  DISENTANGLE: ||A^T B||_F penalty between the two projection maps.
Retrieval mirrors the physics: coarse-filter on HIER, exact-rank on IDENT.
Saves models/hyper_v3/. Usage: train_hyper_v3.py [--epochs N] [--batch N] [--log PATH]
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

DIM_IN, MARGIN = 1024, 1.0
OUT_DIR = "A:/scripts/TheBrain/models/hyper_v3"
TAU = 0.07


class V3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, 512), torch.nn.LayerNorm(512), torch.nn.GELU(),
            torch.nn.Linear(512, 512), torch.nn.LayerNorm(512), torch.nn.GELU(),
            torch.nn.Linear(512, 256), torch.nn.LayerNorm(256), torch.nn.GELU())
        self.proj_h = torch.nn.Linear(256, 64)
        self.proj_i = torch.nn.Linear(256, 64)
        self.drop = torch.nn.Dropout(0.1)
        self.log_scale = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, augment=False):
        t = self.trunk(x)
        if augment:
            t = self.drop(t)
        h = expmap0(self.proj_h(t) * self.log_scale.exp())
        z = torch.nn.functional.normalize(self.proj_i(t), dim=-1)
        return h, z


def info_nce(z1, z2, tau=TAU):
    """SimCLR: views of same anchor attract, all others repel."""
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / tau
    sim.fill_diagonal_(-1e9)
    pos = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    logp = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    return (-logp[torch.arange(2 * n).to(z.device), pos]).mean()


def orth_penalty(model):
    """Subspace disentanglement: projection maps stay orthogonal."""
    A = model.proj_h.weight
    B = model.proj_i.weight
    return ((A.t() @ B) ** 2).mean()


def load_pairs(parent_embs):
    """[(anchor_emb, parent_emb, neg_emb)] — same triplets as v1."""
    import numpy as np
    from core import db as _db
    from core.embeddings import decode_embedding_blob
    conn = _db.db_connect("mapping")
    out = []
    with open("A:/scripts/TheBrain/data/sibling_neg.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                a = conn.execute("SELECT emb FROM entities WHERE canonical_id=?",
                                 (d["anchor"],)).fetchone()
                b = conn.execute("SELECT emb FROM entities WHERE canonical_id=?",
                                 (d["negative"],)).fetchone()
                pe = parent_embs.get(d["positive"].strip())
                ae = decode_embedding_blob(a[0], "") if a and a[0] else None
                be = decode_embedding_blob(b[0], "") if b and b[0] else None
                if ae is not None and be is not None and pe is not None:
                    out.append((ae.astype("float32"), pe.astype("float32"),
                                be.astype("float32")))
            except Exception:
                continue
    conn.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None

    def say(m):
        line = f"[hyperv3] {m}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"device={dev}")
    import numpy as np
    _pl = os.path.join(V1_DIR, "parent_labels.json")
    _pz = os.path.join(V1_DIR, "parent_embs.npz")
    assert os.path.exists(_pl) and os.path.exists(_pz), "v1 parent cache missing"
    _z = np.load(_pz)
    _labels = json.load(open(_pl))
    parent_embs = {l: _z[f"p{i}"] for i, l in enumerate(_labels)}
    trips = load_pairs(parent_embs)
    say(f"triplets: {len(trips)}")
    A = np.stack([t[0] for t in trips])
    P = np.stack([t[1] for t in trips])
    N = np.stack([t[2] for t in trips])
    n = len(A)
    idx = np.random.default_rng(21).permutation(n)
    cut = int(n * 0.9)
    tr, te = idx[:cut], idx[cut:]
    model = V3().to(dev)
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
            ha, za = model(a, augment=False)
            ha2, za2 = model(a, augment=True)
            hp, _ = model(p)
            hn, _ = model(ne)
            loss_h = torch.relu(mobius_dist(ha, hp) - mobius_dist(ha, hn) + MARGIN).mean()
            loss_h = loss_h + 0.5 * torch.relu(angle_at(hp, ha) - cone_aperture(hp)).mean()
            loss_i = info_nce(za, za2)
            loss_o = orth_penalty(model)
            loss = loss_h + loss_i + 0.2 * loss_o
            opt.zero_grad()
            if not torch.isfinite(loss):
                say("non-finite loss, skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(bi)
            nb += len(bi)
        say(f"epoch {ep + 1}/{args.epochs} loss={tot / max(nb, 1):.4f}")
    model.eval()
    with torch.no_grad():
        correct = ent_ok = 0
        m = len(te)
        for i in range(0, m, B):
            bi = te[i:i + B]
            a = torch.from_numpy(A[bi]).to(dev)
            p = torch.from_numpy(P[bi]).to(dev)
            ne = torch.from_numpy(N[bi]).to(dev)
            ha, _ = model(a)
            hp, _ = model(p)
            hn, _ = model(ne)
            correct += int((mobius_dist(ha, hp) < mobius_dist(ha, hn)).sum())
            ent_ok += int((angle_at(hp, ha) <= cone_aperture(hp)).sum())
        # identity separation: mean cosine to nearest non-self neighbor
        # (lower = less sibling interference = decoherence working).
        import numpy as _np
        s = _np.random.default_rng(3).choice(m, size=min(200, m), replace=False)
        va = torch.from_numpy(A[te[s]]).to(dev)
        _, vz = model(va)
        sim = vz @ vz.t()
        sim.fill_diagonal_(-1e9)
        nn_cos = float(sim.max(-1).values.mean())
        raw = torch.from_numpy(A[te[s]]).to(dev)
        raw = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rsim = raw @ raw.t()
        rsim.fill_diagonal_(-1e9)
        raw_nn = float(rsim.max(-1).values.mean())
        metrics = {"eval_n": m, "hier_retrieval": correct / max(m, 1),
                   "hier_entailment": ent_ok / max(m, 1),
                   "ident_nn_cosine": nn_cos, "raw1024_nn_cosine": raw_nn}
    say(f"EVAL {metrics}")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "v3.pt"))
    json.dump(metrics, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=1)
    if logfh:
        logfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
