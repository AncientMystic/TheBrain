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
