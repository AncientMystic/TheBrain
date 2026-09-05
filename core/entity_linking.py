"""
Collective entity linking via global coherence (generic, no doc-specific logic).

Resolves multiple mentions jointly by maximizing coherence of chosen canonicals
in hyperbolic space + mention-candidate similarity. Uses greedy init + hill-climb
local search (bounded iterations). Preserves geometry (distance_matrix), quality
(keeps pairwise fallback when no candidates).
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)


def collective_link(mentions, candidates_fn, embed_fn=None, max_iter=5):
    """Resolve mentions collectively.

    mentions: list of surface strings.
    candidates_fn(mention) -> list of (canonical, embedding or None).
    embed_fn(text) -> embedding or None (for mention-candidate sim when candidate emb missing).
    Returns {mention_idx: chosen_canonical or None}.
    """
    from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance_matrix
    n = len(mentions)
    if n == 0:
        return {}
    cand_lists = []
    for m in mentions:
        try:
            cands = candidates_fn(m) or []
        except Exception:
            cands = []
        cand_lists.append(cands[:10])
    # Greedy init: best mention-candidate sim each (or first candidate)
    choice = {}
    for i, cands in enumerate(cand_lists):
        if not cands:
            choice[i] = None
        else:
            # Prefer candidate with embedding closest to mention embedding when available
            try:
                if embed_fn is not None:
                    me = embed_fn(mentions[i])
                    if me is not None:
                        mh = ensure_hyperbolic(np.asarray(me, dtype=np.float32), space='hyperbolic')
                        best, best_d = cands[0][0], float('inf')
                        for canon, emb in cands:
                            if emb is None:
                                continue
                            ch = ensure_hyperbolic(np.asarray(emb, dtype=np.float32), space='hyperbolic')
                            d = float(np.linalg.norm(mh - ch))
                            if d < best_d:
                                best_d = d
                                best = canon
                        choice[i] = best
                        continue
            except Exception:
                pass
            choice[i] = cands[0][0]
    # Local search: try swaps improving global coherence (sum pairwise sims)
    try:
        for _ in range(max_iter):
            improved = False
            # Build current canonical embeddings map
            cur_embs = {}
            for i, c in choice.items():
                if c is None:
                    continue
                for canon, emb in cand_lists[i]:
                    if canon == c and emb is not None:
                        cur_embs[i] = ensure_hyperbolic(np.asarray(emb, dtype=np.float32), space='hyperbolic')
                        break
            if len(cur_embs) < 2:
                break
            idxs = list(cur_embs.keys())
            mat = np.stack([cur_embs[i] for i in idxs])
            # Pairwise sims baseline
            for i in idxs:
                for canon, emb in cand_lists[i]:
                    if canon == choice[i] or emb is None:
                        continue
                    try:
                        alt = ensure_hyperbolic(np.asarray(emb, dtype=np.float32), space='hyperbolic')
                        # Coherence gain: sum sim(alt, others) - sum sim(cur, others)
                        cur = cur_embs[i]
                        others = [cur_embs[j] for j in idxs if j != i]
                        omat = np.stack(others)
                        from core.hyperbolic import hyperbolic_distance as _hd
                        cur_s = sum(1.0 / (1.0 + float(_hd(cur, o))) for o in others)
                        alt_s = sum(1.0 / (1.0 + float(_hd(alt, o))) for o in others)
                        if alt_s > cur_s + 1e-6:
                            choice[i] = canon
                            improved = True
                            break
                    except Exception:
                        continue
                if improved:
                    break
            if not improved:
                break
    except Exception as e:
        logger.warning(f"Collective search failed: {e}", exc_info=True)
    return choice
