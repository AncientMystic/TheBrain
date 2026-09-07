"""Live 'Curie in Paris' proof (phase 66): S-scoring on the real corpus.

Embeds the query with the pinned RTX mxbai model, ranks the alias pool
with rank_candidates_scored (Paris map position + shard context), prints
the table, and asserts structural invariants (not exact order — order
legitimately depends on derivation state and query ambiguity):
  - every score in [0,1]
  - Paris, Texas (geo:na) scores below every geo:eu candidate
  - Marie Curie is in the top 5
Usage: prove_curie_paris.py
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def main():
    from core.embeddings import get_embeddings_batch
    from core.mapping_lookup import rank_candidates_scored
    from core import db
    q = get_embeddings_batch(["Curie in Paris"], space="hyperbolic")[0]
    assert q is not None and len(q) == 1024, "query embed failed"
    res = rank_candidates_scored(
        "Curie in Paris", mention_emb=q, limit=8,
        shard_weights={"person:hist": 1.0, "geo:eu": 0.8, "geo:na": 0.3,
                       "work:global": 0.1},
        qlat=48.85, qlon=2.35)
    assert res, "empty candidate pool"
    conn = db.db_connect("mapping")
    rows = []
    for cid, s, parts in res:
        d = conn.execute("SELECT display_name, shard_key FROM entities"
                         " WHERE canonical_id=?", (cid,)).fetchone()
        rows.append((cid, d[0], d[1], s, sorted(parts)))
    conn.close()
    for cid, name, shard, s, parts in rows:
        print(f"{s:.3f} {name[:40]:40s} {shard} {parts}", flush=True)
    assert all(0.0 <= s <= 1.0 for _, _, _, s, _ in rows), "score out of range"
    eu = [s for _, _, sh, s, _ in rows if sh == "geo:eu"]
    na = [s for _, _, sh, s, _ in rows if sh == "geo:na"]
    if eu and na:
        assert min(eu) > max(na), "Texas not crushed by horizontal term"
        print("invariant: every geo:eu candidate beats every geo:na one")
    elif eu and not na:
        print("invariant: no geo:na candidate even reached top 8 (stronger)")
    top5 = [c for c, _, _, _, _ in rows[:5]]
    names = [n for _, n, _, _, _ in rows[:5]]
    assert any("Curie" in n for n in names), "Marie Curie missing from top 5"
    print("invariant: Marie Curie in top 5")
    print("prove_curie_paris: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
