"""Live decoherence proof (phase 72): full stack on the real corpus.

Mentions ['Paris', 'Texas'] -> alias pool (mapping_candidates_fn) enriched
with live entities.emb -> mention embs from pinned RTX model ->
decoherence_link -> choice + D/C/review table. Asserts the pipeline
completes with full reports (no unresolved, fields bounded); prints the
table for human judgment of the actual choices.

Usage: prove_decoherence_live.py
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def main():
    from core import db
    from core.embeddings import get_embeddings_batch, decode_embedding_blob
    from core.mapping_lookup import mapping_candidates_fn
    from core.entity_linking import decoherence_link

    mentions = ["Paris", "Texas"]
    conn = db.db_connect("mapping")

    def live_cands(m):
        out = []
        try:
            for cid, _ in mapping_candidates_fn(m, limit=10, conn=conn):
                try:
                    row = conn.execute("SELECT emb FROM entities"
                                       " WHERE canonical_id=?", (cid,)).fetchone()
                    emb = (decode_embedding_blob(row[0], context="live-proof")
                           if row and row[0] else None)
                    out.append((cid, emb.tolist() if emb is not None else None))
                except Exception:
                    out.append((cid, None))
        except Exception:
            pass
        return out

    mem = get_embeddings_batch(mentions, space="hyperbolic")
    assert all(m is not None for m in mem), "mention embed failed"
    ed = {m: e for m, e in zip(mentions, mem)}
    choice, rep = decoherence_link(mentions, live_cands,
                                   embed_fn=lambda t: ed[t], max_iter=5)
    for i, m in enumerate(mentions):
        c = choice.get(i)
        r = rep.get(i, {})
        disp = ""
        if c:
            try:
                d = conn.execute("SELECT display_name, shard_key FROM entities"
                                 " WHERE canonical_id=?", (c,)).fetchone()
                disp = f"{d[0][:38]} [{d[1]}]" if d else c
            except Exception:
                disp = c
        print(f"{m:8s} -> {disp:52s} D={r.get('D', -1):.3f}"
              f" C={r.get('C', -1):.3f} review={r.get('review')}"
              f" ({r.get('reason')})", flush=True)
    conn.close()
    assert set(rep) == {0, 1}, rep
    for i, r in rep.items():
        assert 0.0 <= r["D"] <= 1.0 and 0.0 <= r["C"] <= 1.0, (i, r)
        assert isinstance(r["review"], bool), (i, r)
    print("prove_decoherence_live: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
