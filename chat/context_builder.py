from core import db
import logging
logger = logging.getLogger(__name__)


def _get_doc_display_name(doc_hash):
    """Return document title if available, else filename, else doc_hash."""
    try:
        conn = db.db_connect("index")
        cur = conn.cursor()
        cur.execute("SELECT title, filename FROM documents WHERE file_hash=?", (doc_hash,))
        row = cur.fetchone()
        conn.close()
        if row:
            return row["title"] or row["filename"] or doc_hash
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        pass
    return doc_hash


def _clean_display(text):
    """Display-time cleanup for stored artifacts (hyphen breaks, whitespace)."""
    try:
        from core.text_utils import normalise_text
        return normalise_text(text or "")
    except Exception:
        return (text or "").strip()


def _dedup_facts(facts, limit=25):
    """Drop exact-duplicate facts (normalized text), keep highest confidence first.

    Returns at most `limit` facts. Generic caps on prompt size, never on corpus.
    """
    try:
        import re as _re
        ordered = sorted(facts, key=lambda f: float(f.get("confidence", 0) or 0), reverse=True)
    except Exception:
        ordered = list(facts)
    seen = set()
    out = []
    for f in ordered:
        try:
            key = _re.sub(r"\s+", " ", str(f.get("fact_text", "")).lower()).strip()
        except Exception:
            continue
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(f)
        if len(out) >= limit:
            break
    return out


def _cut_words(text, limit):
    """Truncate to whole words (never mid-word like text[:500] did)."""
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[:cut if cut > 0 else limit].rstrip()


def build_context(facts, summaries=None, chunks=None, conversation_history=None, max_facts=25):
    parts = []
    if conversation_history:
        parts.append(f"<conversation_history>\n{conversation_history}\n</conversation_history>")
    for fact in _dedup_facts(facts or [], limit=max_facts):
        doc_name = fact.get("doc_name", "unknown")
        doc_hash = fact.get("doc_hash")
        if doc_hash:
            doc_name = _get_doc_display_name(doc_hash)
        try:
            conf = float(fact.get("confidence", 0) or 0)
        except Exception:
            conf = 0.0
        text = _clean_display(fact.get("fact_text"))
        span = _clean_display(fact.get("source_span", ""))
        chunk_id = fact.get("chunk_id", None)
        where = f"{doc_name} (chunk {chunk_id})" if chunk_id is not None else doc_name
        parts.append(f"[Fact from {where} | confidence {conf:.2f}] {text} (source: {span})")
    if summaries:
        for s in summaries:
            parts.append(f"[Summary: {s.get('doc_name','')}] {_clean_display(s.get('summary',''))}")
    if chunks:
        doc_name_cache = {}
        for _, _, doc_hash, text in chunks:
            if doc_hash not in doc_name_cache:
                doc_name_cache[doc_hash] = _get_doc_display_name(doc_hash)
            doc_name = doc_name_cache[doc_hash]
            parts.append(f"[Chunk from doc {doc_name}] {_cut_words(_clean_display(text), 500)}")
    return "\n\n".join(parts)
