"""Hyperbolic retrofit head (phase 96): learn 1024-d -> 128-d Poincaré map.

Frozen mxbai embeddings in, hierarchical 128-d ball out. MLP encoder +
expmap0, trained with entailment-cone loss (child inside parent cone,
Ganea aperture) + max-margin contrastive loss against sibling negatives.
Triples come from data/sibling_neg.jsonl (anchor_cid, positive_label,
negative_cid); anchor/neg vectors from mapping.db; parent-label vectors
embedded once via the local ONNX model and cached.

Saves models/hyperbolic_head/{head.pt, parent_embs.npz, metrics.json}.
Usage: train_hyper_head.py [--epochs N] [--batch N] [--log PATH]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, "A:/scripts/TheBrain")

DIM_IN, DIM_OUT, MARGIN = 1024, 128, 1.0
OUT_DIR = "A:/scripts/TheBrain/models/hyperbolic_head"


def mobius_add(u, v, eps=1e-8, floor=1e-4):
    """Möbius addition ( gyrovector ). All ops broadcast-safe.

    floor: antipodal points (u≈-v) drive the denominator to exactly 0
    (1-2+1), whose backward pass overflows to inf. Floor keeps it finite;
    slight bias, zero NaNs (v2 autopsy).
    """
    uv = (u * v).sum(-1, keepdim=True)
    uu = (u * u).sum(-1, keepdim=True)
    vv = (v * v).sum(-1, keepdim=True)
    num = (1 + 2 * uv + vv) * u + (1 - uu) * v
    den = (1 + 2 * uv + uu * vv).clamp(min=floor)
    return num / den


def mobius_dist(u, v, eps=1e-5):
    """Poincaré distance via atanh (finite gradients; arcosh dies at d=0).

    d(u,v) = 2*atanh(||-u⊕v||), norm strictly clamped below 1 so the
    derivative 1/(1-m^2) stays finite. NaN source #1 (v1 autopsy).
    """
    m = mobius_add(-u, v).norm(dim=-1).clamp(max=1 - eps)
    return 2 * torch.atanh(m)


def expmap0(v, eps=1e-8, max_norm=0.999):
    n = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    out = v * ((n / 2).tanh() / n)
    m = out.norm(dim=-1, keepdim=True).clamp(min=eps)
    return out * (torch.minimum(m, torch.full_like(m, max_norm)) / m)


def cone_aperture(p, eps=1e-8):
    import torch
    n = p.norm(dim=-1).clamp(min=eps, max=1 - eps)
    return torch.asin(((1 - n * n) / (n * (1 + n * n).sqrt() + eps)).clamp(-1 + eps, 1 - eps))


def angle_at(p, c, eps=1e-8):
    """Angle at p between (c-p) and p via atan2 (finite at 0 and pi;
    acos dies at both — NaN source #2)."""
    d = c - p
    dot = (d * p).sum(-1)
    cross = (d - dot.unsqueeze(-1) / p.norm(dim=-1, keepdim=True).clamp(min=eps) ** 2 * p).norm(dim=-1)
    return torch.atan2(cross, dot)


class HyperHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, 512), torch.nn.LayerNorm(512), torch.nn.GELU(),
            torch.nn.Linear(512, 256), torch.nn.LayerNorm(256), torch.nn.GELU(),
            torch.nn.Linear(256, DIM_OUT))
        # Learnable temperature: start near origin (healthy gradients),
        # let entailment loss expand outward. Without this, tanh saturates
        # at init and nothing learns (the depth-saturation audit, in model form).
        self.log_scale = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, x):
        return expmap0(self.net(x) * self.log_scale.exp())


def build_parent_cache(log=None):
    """Embed every distinct parent label once. Returns {label: np vec}."""
    import numpy as np
    from core import db as _db
    from core.embeddings import get_embeddings_batch
    labels = set()
    with open("A:/scripts/TheBrain/data/sibling_neg.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                labels.add(json.loads(line)["positive"].strip())
            except Exception:
                continue
    labels = sorted(l for l in labels if l)
    if log:
        log(f"distinct parent labels: {len(labels)}")
    out = {}
    B = 64
    for i in range(0, len(labels), B):
        chunk = labels[i:i + B]
        embs = get_embeddings_batch(chunk, space="euclidean")
        for t, e in zip(chunk, embs):
            if e is not None:
                out[t] = np.asarray(e, dtype=np.float32)
        if log and i % 640 == 0:
            log(f"parent cache {i}/{len(labels)}")
    np.savez_compressed(os.path.join(OUT_DIR, "parent_embs.npz"), **{f"p{i}": v for i, v in enumerate([out[k] for k in sorted(out)])})
    json.dump(sorted(out), open(os.path.join(OUT_DIR, "parent_labels.json"), "w"))
    return out


def load_triplets(parent_embs):
    """[(anchor_emb, parent_emb, neg_emb)] float32 arrays."""
    import numpy as np
    from core import db as _db
    from core.embeddings import decode_embedding_blob
    conn = _db.db_connect("mapping")
    trips = []
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
                    trips.append((ae.astype("float32"), pe.astype("float32"),
                                  be.astype("float32")))
            except Exception:
                continue
    conn.close()
    return trips


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None

    def say(m):
        line = f"[hyperhead] {m}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"device={dev}")
    _pl, _pz = os.path.join(OUT_DIR, "parent_labels.json"), os.path.join(OUT_DIR, "parent_embs.npz")
    if os.path.exists(_pl) and os.path.exists(_pz):
        import numpy as _np0
        _z = _np0.load(_pz)
        _labels = json.load(open(_pl))
        parent_embs = {l: _z[f"p{i}"] for i, l in enumerate(_labels)}
        say(f"parent vectors loaded from disk: {len(parent_embs)}")
    else:
        parent_embs = build_parent_cache(log=say)
    say(f"parent vectors: {len(parent_embs)}")
    trips = load_triplets(parent_embs)
    say(f"triplets: {len(trips)}")
    assert len(trips) > 1000, "too few triplets"
    import numpy as np
    A = np.stack([t[0] for t in trips])
    P = np.stack([t[1] for t in trips])
    N = np.stack([t[2] for t in trips])
    n = len(A)
    idx = np.random.default_rng(7).permutation(n)
    cut = int(n * 0.9)
    tr, te = idx[:cut], idx[cut:]
    model = HyperHead().to(dev)
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
            ha, hp, hn = model(a), model(p), model(ne)
            d_pos = mobius_dist(ha, hp)
            d_neg = mobius_dist(ha, hn)
            loss_con = torch.relu(d_pos - d_neg + MARGIN).mean()
            ap = cone_aperture(hp)
            ang = angle_at(hp, ha)
            loss_ent = torch.relu(ang - ap).mean()
            loss = loss_con + 0.5 * loss_ent
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
    metrics = {}
    with torch.no_grad():
        correct = ent_ok = 0
        m = len(te)
        for i in range(0, m, B):
            bi = te[i:i + B]
            a = torch.from_numpy(A[bi]).to(dev)
            p = torch.from_numpy(P[bi]).to(dev)
            ne = torch.from_numpy(N[bi]).to(dev)
            ha, hp, hn = model(a), model(p), model(ne)
            correct += int((mobius_dist(ha, hp) < mobius_dist(ha, hn)).sum())
            ent_ok += int((angle_at(hp, ha) <= cone_aperture(hp)).sum())
        metrics = {"eval_n": m, "retrieval_acc": correct / max(m, 1),
                   "entailment_rate": ent_ok / max(m, 1)}
    say(f"EVAL {metrics}")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "head.pt"))
    json.dump(metrics, open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=1)
    if logfh:
        logfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
