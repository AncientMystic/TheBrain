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
