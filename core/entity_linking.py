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


# Decoherence-weighted resolution (phase 69): quantum-decoherence model of
# disambiguation. Candidates start in superposition (prior weights); the
# document context (other mentions' choices) is the environment. Choices
# that correlate distinctly with the environment decohere fast (high D,
# accept cheaply); ambiguous ones stay coherent (low C, route to the
# Enneagram shock gate = human review instead of burning verify compute).
# The hill-climb itself is ms-scale — the compute saved is downstream
# verification, which this report gates. Additive; collective_link intact.
def decoherence_report(mentions, choice, candidates_fn, embed_fn=None,
                       gamma=1.0, review_threshold=0.35, tie_eps=0.02):
    """Per-mention {D, C, review}: distinguishability, coherence, gate flag.

    D_i = environmental margin: mean similarity of the chosen emb to
    other resolved choices minus the best unchosen alternative's (how
    well the environment separates the alternatives; single-candidate
    or single-mention cases default to 1.0). A_i = ambiguity = tied
    candidates within tie_eps of the top mention-candidate sim, normalized.
    C_i = exp(-gamma * A_i) (coherence decay). review_i = C or D below
    threshold. Never raises; missing embeddings degrade to neutral.
    """
    from core.hyperbolic import ensure_hyperbolic, hyperbolic_similarity
    rep = {}
    try:
        n = len(mentions)
        chosen_embs = {}
        for i, c in (choice or {}).items():
            if c is None:
                continue
            try:
                for canon, emb in (candidates_fn(mentions[i]) or []):
                    if canon == c and emb is not None:
                        chosen_embs[i] = ensure_hyperbolic(
                            np.asarray(emb, dtype=np.float32),
                            space='hyperbolic')
                        break
            except Exception:
                continue
        for i in range(n):
            c = (choice or {}).get(i)
            if c is None or i not in chosen_embs:
                rep[i] = {"D": 0.0, "C": 0.0, "review": True,
                          "reason": "unresolved"}
                continue
            others = [e for j, e in chosen_embs.items() if j != i]
            if others:
                try:
                    sup = lambda e: float(np.mean(
                        [hyperbolic_similarity(e, o) for o in others]))
                    sup_hit = sup(chosen_embs[i])
                    sup_alt = None
                    try:
                        for canon, emb in (candidates_fn(mentions[i]) or [])[:10]:
                            if canon == c or emb is None:
                                continue
                            a = ensure_hyperbolic(
                                np.asarray(emb, dtype=np.float32),
                                space='hyperbolic')
                            s = sup(a)
                            sup_alt = s if sup_alt is None else max(sup_alt, s)
                    except Exception:
                        pass
                    if sup_alt is None:
                        D = 1.0  # single candidate: nothing to distinguish
                    elif sup_alt <= sup_hit:
                        # choice is env-best: D rewards the margin, floored
                        # at 0.5 so coherent-but-close pairs still pass
                        D = min(max(0.5 + (sup_hit - sup_alt), 0.0), 1.0)
                    else:
                        D = 0.0
                except Exception:
                    D = 0.5
            else:
                D = 1.0
            A = 0.0
            try:
                sims = []
                me = embed_fn(mentions[i]) if embed_fn else None
                mh = ensure_hyperbolic(np.asarray(me, dtype=np.float32),
                                       space='hyperbolic') if me is not None else None
                for canon, emb in (candidates_fn(mentions[i]) or [])[:10]:
                    if emb is None or mh is None:
                        continue
                    ch = ensure_hyperbolic(np.asarray(emb, dtype=np.float32),
                                           space='hyperbolic')
                    sims.append(hyperbolic_similarity(mh, ch))
                if len(sims) > 1:
                    top = max(sims)
                    tied = sum(1 for s in sims if top - s <= tie_eps) - 1
                    A = min(max(tied / max(len(sims) - 1, 1), 0.0), 1.0)
            except Exception:
                pass
            try:
                C = float(np.exp(-float(gamma) * A))
            except Exception:
                C = 0.5
            D = min(max(float(D), 0.0), 1.0)
            review = bool(C < review_threshold or D < review_threshold)
            rep[i] = {"D": D, "C": C, "review": review,
                      "reason": "ok" if not review else
                      ("ambiguous" if C < review_threshold else "indistinct")}
    except Exception as e:
        logger.warning(f"Decoherence report failed: {e}", exc_info=True)
    return rep


def decoherence_link(mentions, candidates_fn, embed_fn=None, max_iter=5,
                     gamma=1.0, review_threshold=0.35):
    """collective_link + decoherence gating. Returns (choice, report).

    choice: {idx: canonical or None} (unchanged semantics).
    report: {idx: {D, C, review, reason}} — review=True means route to the
    shock gate (human/LLM review), not auto-accept. Backward compatible:
    existing callers ignoring the second value keep working.
    """
    choice = collective_link(mentions, candidates_fn, embed_fn=embed_fn,
                             max_iter=max_iter)
    report = decoherence_report(mentions, choice, candidates_fn,
                                embed_fn=embed_fn, gamma=gamma,
                                review_threshold=review_threshold)
    return choice, report
