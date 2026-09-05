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
    """Drop exact-duplicate facts (normalized text), keep highest confidence each.

    Output follows FIRST-retrieval order (ranker intent survives); the best
    duplicate wins its slot. Deterministic for identical inputs. Generic caps
    on prompt size, never on corpus.
    """
    import re as _re
    best = {}
    first_at = {}
    order = []
    for i, f in enumerate(facts or []):
        try:
            key = _re.sub(r"\s+", " ", str(f.get("fact_text", "")).lower()).strip()
        except Exception:
            continue
        if not key:
            continue
        if key not in best:
            best[key] = f
            first_at[key] = i
            order.append(key)
            continue
        try:
            if float(f.get("confidence", 0) or 0) > float(best[key].get("confidence", 0) or 0):
                best[key] = f
        except Exception:
            pass
    out = [best[k] for k in sorted(order, key=lambda k: first_at[k])]
    return out[:max(1, limit)] if out else []


def _cut_words(text, limit):
    """Truncate to whole words (never mid-word like text[:500] did)."""
    text = text or ""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[:cut if cut > 0 else limit].rstrip()


def build_context(facts, summaries=None, chunks=None, conversation_history=None, max_facts=25,
                  budget_chars=None, model_label=""):
    parts = []
    if conversation_history:
        parts.append(f"<conversation_history>\n{conversation_history}\n</conversation_history>")
    kept = _dedup_facts(facts or [], limit=max_facts)
    omitted_facts = 0
    if budget_chars is not None:
        try:
            room = max(0, int(budget_chars) - sum(len(p) for p in parts))
            fitted, used = [], 0
            for fact in kept:
                est = len(str(fact.get("fact_text", ""))) + len(str(fact.get("source_span", ""))) + 120
                if fitted and used + est > room:
                    continue
                fitted.append(fact)
                used += est
            if not fitted and kept:
                fitted = kept[:1]
            omitted_facts = len(kept) - len(fitted)
            kept = fitted
        except Exception:
            pass
    for fact in kept:
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
    omitted_chunks = 0
    if chunks:
        doc_name_cache = {}
        room = None
        if budget_chars is not None:
            try:
                room = max(0, int(budget_chars) - sum(len(p) for p in parts))
            except Exception:
                room = None
        for _, _, doc_hash, text in chunks:
            if doc_hash not in doc_name_cache:
                doc_name_cache[doc_hash] = _get_doc_display_name(doc_hash)
            doc_name = doc_name_cache[doc_hash]
            line = f"[Chunk from doc {doc_name}] {_cut_words(_clean_display(text), 500)}"
            if room is not None and room - len(line) < 0:
                omitted_chunks += 1
                continue
            parts.append(line)
            if room is not None:
                room -= len(line)
    if budget_chars is not None and (omitted_facts or omitted_chunks):
        try:
            label = model_label or "current model"
            parts.append(f"[Context fit to {label}: {omitted_facts} lower-ranked facts and "
                         f"{omitted_chunks} excerpts omitted; ranked highest-first. "
                         f"Answer from what is shown; say so if it is insufficient.]")
        except Exception:
            pass
    return "\n\n".join(parts)


def _fact_line(tag, fact, doc_name_cache):
    """Single tagged fact line. Tag may be None for untagged legacy lines."""
    doc_name = fact.get("doc_name", "unknown")
    doc_hash = fact.get("doc_hash")
    if doc_hash:
        if doc_hash not in doc_name_cache:
            doc_name_cache[doc_hash] = _get_doc_display_name(doc_hash)
        doc_name = doc_name_cache[doc_hash]
    try:
        conf = float(fact.get("confidence", 0) or 0)
    except Exception:
        conf = 0.0
    text = _clean_display(fact.get("fact_text"))
    span = _clean_display(fact.get("source_span", ""))
    chunk_id = fact.get("chunk_id", None)
    where = f"{doc_name} (chunk {chunk_id})" if chunk_id is not None else doc_name
    head = f"[{tag}] " if tag else ""
    return f"{head}[Fact from {where} | confidence {conf:.2f}] {text} (source: {span})"


def build_tagged_context(facts, summaries=None, chunks=None, conversation_history=None,
                         max_facts=25, budget_chars=None, model_label="", tag_prefix="S"):
    """Tagged context with ONE ordering authority shared by text and payload.

    Returns (text, ordered, tagmap) where ordered[i] carries tag S{i+1} and
    tagmap maps tag -> {doc, confidence}. Dedup keeps best-confidence text in
    first-retrieval position; budget fit preserves that order; numbering follows
    final order so frontend footnotes numbered in received order always align.
    Deterministic for identical inputs.
    """
    parts = []
    if conversation_history:
        parts.append(f"<conversation_history>\n{conversation_history}\n</conversation_history>")
    ordered_all = _dedup_facts(facts or [], limit=max_facts)
    ordered, used = [], 0
    room = None
    if budget_chars is not None:
        try:
            room = max(0, int(budget_chars) - sum(len(p) for p in parts))
        except Exception:
            room = None
    omitted_facts = 0
    for fact in ordered_all:
        est = len(str(fact.get("fact_text", ""))) + len(str(fact.get("source_span", ""))) + 140
        if room is not None and ordered and room - est < 0:
            omitted_facts += 1
            continue
        ordered.append(fact)
        if room is not None:
            room -= est
    if not ordered and ordered_all:
        ordered = ordered_all[:1]
        omitted_facts = max(0, omitted_facts - 1)
    doc_name_cache = {}
    tagmap = {}
    for i, fact in enumerate(ordered):
        tag = f"{tag_prefix}{i + 1}"
        try:
            fact["citation_tag"] = tag
        except Exception:
            pass
        parts.append(_fact_line(tag, fact, doc_name_cache))
        try:
            tagmap[tag] = {"doc": fact.get("doc_name", "unknown"),
                           "confidence": float(fact.get("confidence", 0) or 0)}
            if fact.get("doc_hash"):
                tagmap[tag]["doc"] = doc_name_cache.get(fact["doc_hash"], tagmap[tag]["doc"])
        except Exception:
            pass
    omitted_chunks = 0
    if chunks:
        for _, _, doc_hash, text in chunks:
            if doc_hash not in doc_name_cache:
                doc_name_cache[doc_hash] = _get_doc_display_name(doc_hash)
            line = f"[Chunk from doc {doc_name_cache[doc_hash]}] {_cut_words(_clean_display(text), 500)}"
            if room is not None and room - len(line) < 0:
                omitted_chunks += 1
                continue
            parts.append(line)
            if room is not None:
                room -= len(line)
    if summaries:
        for s in summaries:
            parts.append(f"[Summary: {s.get('doc_name','')}] {_clean_display(s.get('summary',''))}")
    if budget_chars is not None and (omitted_facts or omitted_chunks):
        try:
            label = model_label or "current model"
            parts.append(f"[Context fit to {label}: {omitted_facts} lower-ranked facts and "
                         f"{omitted_chunks} excerpts omitted; ranked highest-first. "
                         f"Answer from what is shown; say so if it is insufficient.]")
        except Exception:
            pass
    return "\n\n".join(parts), ordered, tagmap


def build_tag_ledger(ordered_facts, max_chars=2000):
    """Compact tag ledger for free-form (grouped) contexts: tag -> doc (conf).

    Appended after organized narratives whose lines can't carry tags themselves.
    """
    lines = ["[Sources]"]
    used = 0
    for fact in ordered_facts or []:
        tag = fact.get("citation_tag") if isinstance(fact, dict) else None
        if not tag:
            continue
        try:
            doc = str(fact.get("doc_name", fact.get("doc_hash", "unknown")))
            conf = float(fact.get("confidence", 0) or 0)
            first = str(fact.get("fact_text", ""))[:80]
        except Exception:
            continue
        line = f"{tag}: {doc} (conf {conf:.2f}) — {first}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if len(lines) > 1 else ""
