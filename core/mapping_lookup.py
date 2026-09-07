"""
Mapping lookup: candidate generation for collective entity linking (P4).

Implements the missing `candidates_fn` for core.entity_linking.collective_link:
mention text -> [(canonical_id, emb_or_None)], ranked exact-alias first,
FTS fallback second, capped. Embeddings stay None here — the caller
supplies embed_fn (mxbai) per the linker contract. Read-only; never writes.
"""
import re

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_mention(text):
    """Lowercase alphanumeric token join: 'Marie Curie' -> 'marie curie'."""
    return " ".join(_WORD_RE.findall(str(text or "").lower()))


def mapping_candidates_fn(mention_text, limit=10, conn=None):
    """Return [(canonical_id, None-emb)] candidates for a mention surface."""
    norm = normalize_mention(mention_text)
    if not norm:
        return []
    own_conn = False
    if conn is None:
        from core import db as _db
        conn = _db.db_connect("mapping")
        own_conn = True
    try:
        seen, out = set(), []
        try:
            for row in conn.execute(
                    "SELECT canonical_id FROM aliases WHERE alias_norm=? LIMIT ?",
                    (norm, int(limit))):
                cid = row[0]
                if cid not in seen:
                    seen.add(cid)
                    out.append((cid, None))
        except Exception:
            pass
        if len(out) < int(limit):
            try:
                toks = [t for t in norm.split() if len(t) > 2]
                if toks:
                    q = " OR ".join(f'"{t}"*' for t in toks[:6])
                    for row in conn.execute(
                            "SELECT canonical_id FROM aliases_fts WHERE aliases_fts MATCH ? LIMIT ?",
                            (q, int(limit))):
                        cid = row[0]
                        if cid not in seen:
                            seen.add(cid)
                            out.append((cid, None))
                        if len(out) >= int(limit):
                            break
            except Exception:
                pass
        return out[:int(limit)]
    finally:
        if own_conn:
            try:
                conn.close()
            except Exception:
                pass


def mapping_context_block(ent_names, max_chars=1200, conn=None):
    """Build a compact 'Known:' context block for chat prompts.

    Resolves each entity surface via mapping_candidates_fn (first hit),
    then formats display cards as one line each. Bounded, read-only,
    never raises. The server chat path appends this to context so the
    AI knows who/where each name is without bloating the prompt.
    """
    try:
        names = []
        for _e in (ent_names or []):
            _t = _e.get("text") if isinstance(_e, dict) else str(_e)
            if _t and _t.strip():
                names.append(_t.strip())
        if not names:
            return ""
        own_conn = False
        if conn is None:
            from core import db as _db
            conn = _db.db_connect("mapping")
            own_conn = True
        try:
            lines = []
            for _n in names[:12]:
                try:
                    _cands = mapping_candidates_fn(_n, limit=1, conn=conn)
                except Exception:
                    continue
                if not _cands:
                    continue
                _card = lookup_display(conn, _cands[0][0])
                if not _card or not _card.get("display_name"):
                    continue
                _line = f"Known: {_card['display_name']} [{_card.get('entity_type', '')}]"
                if _card.get("region_code"):
                    _line += f" ({_card['region_code']})"
                if _card.get("description"):
                    _line += f" — {_card['description'][:160]}"
                lines.append(_line)
            block = "\n".join(lines)
            if len(block) > int(max_chars):
                block = block[:int(max_chars)] + "…"
            return block
        finally:
            if own_conn:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        return ""


def record_dirty_mention(conn, surface, doc_id, proposed_canonical=""):
    """Queue an unlinked/low-coherence mention for human/LLM review.

    The repair queue behind clean canonical mapping: approvers promote
    rows to aliases/same_as edges; nothing rewrites legacy tables in
    place. Idempotent, never raises.
    """
    try:
        if not (surface or "").strip() or not (doc_id or "").strip():
            return False
        conn.execute("INSERT OR IGNORE INTO dirty_mentions (surface, doc_id, proposed_canonical, status) VALUES (?,?,?,?)",
                     (surface.strip(), doc_id.strip(), proposed_canonical or "", "pending"))
        conn.commit()
        return True
    except Exception:
        return False


def lookup_display(conn, canonical_id):
    """One-row display card for chat injection: name, type, description."""
    try:
        row = conn.execute(
            "SELECT display_name, entity_type, description, region_code, popularity"
            " FROM entities WHERE canonical_id=?", (canonical_id,)).fetchone()
        if not row:
            return None
        return {"display_name": row[0], "entity_type": row[1],
                "description": row[2] or "", "region_code": row[3] or "",
                "popularity": row[4] or 0}
    except Exception:
        return None


# Hyper-coordinate relevance weights (phase 65). S_hyp carries the semantic
# match, S_plane the linear map pre-filter term, S_shard the horizontal term.
W_HYP, W_PLANE, W_SHARD = 0.6, 0.25, 0.15


def _plane_similarity(clat, clon, qlat, qlon):
    """Equirectangular degree distance -> (0,1] similarity, 45deg scale."""
    try:
        import math
        if None in (clat, clon, qlat, qlon):
            return None
        mlat = math.radians((float(clat) + float(qlat)) / 2.0)
        dd = math.hypot(float(clat) - float(qlat),
                        math.cos(mlat) * (float(clon) - float(qlon)))
        return 1.0 / (1.0 + dd / 45.0)
    except Exception:
        return None


def rank_candidates_scored(mention_text, mention_emb=None, limit=10,
                           shard_weights=None, qlat=None, qlon=None,
                           conn=None):
    """Rank alias candidates by S = 0.6*S_hyp + 0.25*S_plane + 0.15*S_shard.

    mention_emb: hyperbolic query vector (list/np) or None. shard_weights:
    {shard_key: 0..1} from document context (None = no shard signal).
    qlat/qlon: query map position or None. Missing parts are EXCLUDED and
    the remaining weights renormalize (graceful degradation, never 0-bias).
    Returns [(canonical_id, S, parts)] sorted desc. Read-only, never raises.
    """
    try:
        limit = max(int(limit), 1)
        own_conn = False
        if conn is None:
            from core import db as _db
            conn = _db.db_connect("mapping")
            own_conn = True
        try:
            pool = mapping_candidates_fn(mention_text, limit=min(limit * 3, 30),
                                         conn=conn)
            if not pool:
                return []
            try:
                from core.hyperbolic import hyperbolic_similarity
                import numpy as _np
                _have_hyp = True
            except Exception:
                _have_hyp = False
            q = None
            if mention_emb is not None and _have_hyp:
                try:
                    q = _np.asarray(mention_emb, dtype=_np.float64)
                except Exception:
                    q = None
            scored = []
            for order, (cid, _) in enumerate(pool):
                try:
                    row = conn.execute(
                        "SELECT emb, lat_r, lon_r, shard_key FROM entities"
                        " WHERE canonical_id=?", (cid,)).fetchone()
                except Exception:
                    row = None
                parts, num, den = {}, 0.0, 0.0
                if row:
                    blob, clat, clon, shard = row[0], row[1], row[2], row[3]
                    if q is not None and blob is not None and _have_hyp:
                        try:
                            import numpy as _np2
                            c = _np2.frombuffer(bytes(blob), dtype=_np2.float32)
                            if len(c) == len(q):
                                s = hyperbolic_similarity(q, c)
                                parts["hyp"] = s
                                num += W_HYP * s
                                den += W_HYP
                        except Exception:
                            pass
                    ps = _plane_similarity(clat, clon, qlat, qlon)
                    if ps is not None:
                        parts["plane"] = ps
                        num += W_PLANE * ps
                        den += W_PLANE
                    if shard_weights and shard is not None and shard in shard_weights:
                        try:
                            s = min(max(float(shard_weights[shard]), 0.0), 1.0)
                            parts["shard"] = s
                            num += W_SHARD * s
                            den += W_SHARD
                        except Exception:
                            pass
                score = (num / den) if den > 0 else 0.5
                scored.append((cid, score, parts, order))
            scored.sort(key=lambda t: (-t[1], t[3]))
            return [(cid, s, p) for cid, s, p, _ in scored[:limit]]
        finally:
            if own_conn:
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception:
        return []
