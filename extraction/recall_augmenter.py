"""
DB-aware recall augmenter for priority-guaranteed extraction.

Scans preloaded RecallIndex during fast pass (no LLM) to match similar
topics, dates, references, events and flag priority to ensure capture.
Priority = must-extract + must-verify (escalate), never must-believe.
Preserves hyperbolic geometry (distance, not cosine), verification-first,
no lazy truncation, no doc-specific hardcoding (weights/thresholds in config).
"""
import re
import config
import logging
logger = logging.getLogger(__name__)

_YEAR_RE = re.compile(r'\b(17|18|19|20)\d{2}\b')


def _default_weights():
    return {
        "entity": 0.3,
        "topic": 0.2,
        "date": 0.25,
        "event": 0.15,
        "standards": 0.3,
        "contradiction": 0.5,
    }


def augment_batch(chunks, fast_pres=None, chunk_embs=None, recall_index=None):
    """Return list of recall dicts per chunk: {score, priority, reasons, context}.

    Batched (single FTS + single distance_matrix per batch), no per-chunk DB storm.
    """
    from extraction.recall_index import RecallIndex
    idx = recall_index or RecallIndex.load()
    n = len(chunks)
    out = [{"score": 0.0, "priority": False, "reasons": [], "context": "", "hits": {}} for _ in range(n)]
    if n == 0:
        return out
    try:
        weights = dict(_default_weights())
        cfg_w = getattr(config, "RECALL_PRIORITY_WEIGHTS", None)
        if isinstance(cfg_w, dict):
            for k, v in cfg_w.items():
                if k in weights:
                    try:
                        weights[k] = float(v)
                    except Exception:
                        pass
        threshold = float(getattr(config, "RECALL_PRIORITY_THRESHOLD", 0.5))
    except Exception:
        weights = _default_weights()
        threshold = 0.5

    # Pre-tokenize + fast entities per chunk (reuse fast_pres when available)
    chunk_entity_texts = []
    for i, ch in enumerate(chunks):
        try:
            if fast_pres and i < len(fast_pres) and fast_pres[i]:
                pre = fast_pres[i]
                ents = []
                for k in ("entities", "people", "locations", "organizations"):
                    for it in pre.get(k, []) or []:
                        if isinstance(it, dict):
                            t = it.get("text") or it.get("entity_name") or it.get("person_name") or it.get("location_name") or ""
                            if t:
                                ents.append(str(t))
                chunk_entity_texts.append(ents)
            else:
                chunk_entity_texts.append([])
        except Exception:
            chunk_entity_texts.append([])

    # Entity linking via automaton with word boundaries + collective coherence prune.
    # Single batched embedding fetch for all candidates (no per-chunk HTTP storm).
    linked = [[] for _ in range(n)]
    _all_cands = []
    try:
        A = idx.alias_automaton
        if A is not None:
            for i, ch in enumerate(chunks):
                try:
                    low = ch.lower()
                    # mention -> list[canon] (ambiguity preserved); single-canon legacy shape supported
                    mentions = {}
                    for end_idx, (canon_val, term_lower) in A.iter(low):
                        try:
                            start_idx = end_idx - len(term_lower) + 1
                            if start_idx > 0 and ch[start_idx - 1].isalnum():
                                continue
                            if end_idx + 1 < len(ch) and ch[end_idx + 1].isalnum():
                                continue
                            canons = canon_val if isinstance(canon_val, list) else [canon_val]
                            key = (start_idx, end_idx)
                            if key not in mentions:
                                mentions[key] = []
                            for _c in canons:
                                if _c not in mentions[key] and len(mentions[key]) < 5:
                                    mentions[key].append(_c)
                            if len(mentions) >= 20:
                                break
                        except Exception:
                            continue
                    # Collective choice per chunk across ambiguous mentions (generic coherence)
                    cands = []
                    try:
                        if mentions and any(len(v) > 1 for v in mentions.values()):
                            from core.entity_linking import collective_link as _cl2
                            _mkeys = list(mentions.keys())
                            _mnames = [f"m{k}" for k in _mkeys]
                            _cands_fn = lambda m, _mk=_mkeys, _mn=_mnames, _mm=mentions: (
                                [(c, None) for c in _mm[_mk[_mn.index(m)]]] if m in _mn else [])
                            _res = _cl2(_mnames, _cands_fn)
                            for mk, mn in zip(_mkeys, _mnames):
                                _chosen = _res.get(mn) if isinstance(_res, dict) else None
                                if _chosen:
                                    cands.append(_chosen)
                                else:
                                    cands.extend(mentions[mk][:1])
                        else:
                            for v in mentions.values():
                                cands.extend(v[:1])
                    except Exception:
                        for v in mentions.values():
                            cands.extend(v[:1])
                    # Dedup preserve order, cap 20
                    seen = {}
                    for _c in cands:
                        if _c not in seen and len(seen) < 20:
                            seen[_c] = True
                    cands = list(seen.keys())[:20]
                    linked[i] = cands
                    _all_cands.extend(cands)
                except Exception:
                    continue
            # Single batch for all unique candidates
            _uniq = list(dict.fromkeys(_all_cands))
            _emap = {}
            if _uniq:
                try:
                    from core.embeddings import get_embeddings_dict as _ged
                    _emap = _ged(_uniq, space='hyperbolic')
                except Exception:
                    _emap = {}
            # Per-chunk coherence prune using cached vectors (no HTTP in loop)
            try:
                from core.hyperbolic import ensure_hyperbolic as _eh, hyperbolic_distance_matrix as _dm
                import numpy as _np3
                for i in range(n):
                    cands = linked[i]
                    if len(cands) <= 4:
                        continue
                    vecs = []
                    valid = []
                    for c in cands:
                        e = _emap.get(c)
                        if e is None:
                            valid.append(False)
                        else:
                            try:
                                vecs.append(_eh(_np3.asarray(e, dtype=_np3.float32), space='hyperbolic'))
                                valid.append(True)
                            except Exception:
                                valid.append(False)
                    if sum(valid) < 3 or not vecs:
                        linked[i] = cands[:10]
                        continue
                    try:
                        pmat = _np3.stack(vecs)
                        dmat = _dm(pmat, pmat)
                        sims = 1.0 / (1.0 + dmat)
                        _np3.fill_diagonal(sims, 0.0)
                        means = sims.mean(axis=1)
                        med = float(_np3.median(means))
                        kept = []
                        vi = 0
                        for c, v in zip(cands, valid):
                            if not v:
                                kept.append(c)
                                continue
                            try:
                                if float(means[vi]) >= med:
                                    kept.append(c)
                                vi += 1
                            except Exception:
                                vi += 1
                                kept.append(c)
                        if len(kept) < 4:
                            kept = cands[:4]
                        linked[i] = kept[:10]
                    except Exception:
                        linked[i] = cands[:10]
            except Exception:
                pass
    except Exception:
        pass

    # Topic sims vectorized (single matrix when centroids + embs available)
    topic_sims = [0.0] * n
    try:
        if idx.topic_centroids and chunk_embs:
            import numpy as _np
            from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance_matrix
            cents = [c for _, c in idx.topic_centroids[:20]]
            cmat = _np.stack([ensure_hyperbolic(c, space='hyperbolic') for c in cents])
            qlist = []
            qidx = []
            for i, e in enumerate(chunk_embs):
                if e is None or i >= n:
                    continue
                try:
                    qlist.append(ensure_hyperbolic(_np.asarray(e, dtype=_np.float32), space='hyperbolic'))
                    qidx.append(i)
                except Exception:
                    continue
            if qlist:
                qmat = _np.stack(qlist)
                dmat = hyperbolic_distance_matrix(qmat, cmat)
                for r, i in enumerate(qidx):
                    try:
                        best = float(_np.min(dmat[r]))
                        topic_sims[i] = float(1.0 / (1.0 + best))
                    except Exception:
                        pass
    except Exception:
        pass

    # Standards / contradiction sims batched (single matrix for all chunks with embs)
    std_sims = [0.0] * n
    std_neg_mismatch = [False] * n
    try:
        if idx.standards and chunk_embs:
            import numpy as _np2
            from core.hyperbolic import ensure_hyperbolic as _eh, hyperbolic_distance_matrix as _dm
            smat = _np2.stack([_eh(_np2.asarray(s["embedding"], dtype=_np2.float32), space='hyperbolic') for s in idx.standards if s.get("embedding") is not None])
            qlist = []
            qidx = []
            for i, e in enumerate(chunk_embs):
                if e is None or i >= n:
                    continue
                try:
                    qlist.append(_eh(_np2.asarray(e, dtype=_np2.float32), space='hyperbolic'))
                    qidx.append(i)
                except Exception:
                    continue
            if qlist and len(smat):
                qmat = _np2.stack(qlist)
                dmat = _dm(qmat, smat)
                for r, i in enumerate(qidx):
                    try:
                        bi = int(_np2.argmin(dmat[r]))
                        best_d = float(dmat[r][bi])
                        std_sims[i] = float(1.0 / (1.0 + best_d))
                        # Negation mismatch heuristic: chunk mentions "not/no/never" near linked entity
                        # while standard negation differs -> contradiction candidate (must-verify)
                        low = chunks[i].lower()
                        has_neg = (" not " in low or " no " in low or " never " in low or "n't " in low)
                        std_neg = idx.standards[bi].get("negation", 0) or 0
                        # We don't know chunk negation yet; flag when high sim + neg words present
                        if std_sims[i] > 0.6 and has_neg:
                            std_neg_mismatch[i] = True
                    except Exception:
                        continue
    except Exception:
        pass

    for i, ch in enumerate(chunks):
        reasons = []
        score = 0.0
        hits = {}
        try:
            # Entity
            ents = linked[i] or chunk_entity_texts[i]
            if ents:
                hits["entities"] = ents[:5]
                score += weights["entity"] * min(1.0, len(ents) / 3.0)
                reasons.append(f"linked {len(ents)} known entities")
            # Topic
            if topic_sims[i] > 0.4:
                hits["topic_sim"] = round(topic_sims[i], 3)
                score += weights["topic"] * topic_sims[i]
                reasons.append(f"topic sim {topic_sims[i]:.2f}")
            # Dates
            try:
                years = set(m.group() for m in _YEAR_RE.finditer(ch))
                dhit = years & idx.date_anchors if hasattr(idx.date_anchors, "__contains__") else set()
                if dhit:
                    hits["dates"] = sorted(dhit)[:5]
                    score += weights["date"]
                    reasons.append(f"date anchors {sorted(dhit)[:3]}")
            except Exception:
                pass
            # Events
            try:
                low = ch.lower()
                ev = [t for t in idx.event_triggers if t and t in low]
                if ev:
                    hits["events"] = ev[:5]
                    score += weights["event"] * min(1.0, len(ev) / 2.0)
                    reasons.append(f"event triggers {ev[:3]}")
            except Exception:
                pass
            # Standards
            if std_sims[i] > 0.5:
                hits["standards_sim"] = round(std_sims[i], 3)
                score += weights["standards"] * std_sims[i]
                reasons.append(f"standards sim {std_sims[i]:.2f}")
            if std_neg_mismatch[i]:
                hits["contradiction_candidate"] = True
                score += weights["contradiction"]
                reasons.append("possible contradiction of anchor (must-verify)")
            # Hard anchors (always priority, no weighting needed)
            hard = False
            try:
                norm = re.sub(r'\s+', ' ', ch.lower()).strip()
                for s in idx.standards[:200]:
                    ss = str(s.get("statement", "")).lower().strip()
                    if ss and (ss in norm or norm in ss):
                        hard = True
                        reasons.append("exact anchor match")
                        break
            except Exception:
                pass
            priority = hard or (score >= threshold)
            # Build concise recall_context for prompt (<=500ch, templated)
            ctx_parts = []
            if hits.get("entities"):
                ctx_parts.append("Linked: " + ", ".join(hits["entities"][:5]))
            if hits.get("dates"):
                ctx_parts.append("Dates: " + ", ".join(hits["dates"][:5]))
            if hits.get("events"):
                ctx_parts.append("Events: " + ", ".join(hits["events"][:5]))
            if hits.get("standards_sim"):
                ctx_parts.append(f"See standards (sim {hits['standards_sim']}) for contradictions")
            if hits.get("contradiction_candidate"):
                ctx_parts.append("Check negation vs anchor carefully")
            context = "; ".join(ctx_parts)[:500]
            out[i] = {"score": round(float(score), 3), "priority": bool(priority), "reasons": reasons[:5], "context": context, "hits": hits}
        except Exception:
            continue
    return out
