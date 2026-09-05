"""
Character-offset span validation with nearest-sentence fallback (generic).

Checks exact substring presence, then char offsets when provided, then falls
back to nearest sentence containing fact keywords. Never fabricates spans;
returns (valid, corrected_span, reason). Preserves provenance for verifier.
"""
import re


def _split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    out = []
    pos = 0
    for p in parts:
        start = text.find(p, pos)
        if start < 0:
            start = pos
        out.append((p, start, start + len(p)))
        pos = start + len(p)
    return out


def validate_span(fact_text, source_span, chunk_text, start_char=None, end_char=None):
    """Validate span exists in chunk. Returns (valid, span, reason)."""
    chunk = chunk_text or ""
    span = (source_span or "").strip()
    if span and span in chunk:
        if start_char is not None and end_char is not None:
            try:
                if chunk[int(start_char):int(end_char)] == span:
                    return True, span, "exact+offsets"
            except Exception:
                pass
        return True, span, "exact"
    if start_char is not None and end_char is not None:
        try:
            s, e = int(start_char), int(end_char)
            cand = chunk[s:e]
            if cand and cand in chunk:
                return True, cand, "offsets"
        except Exception:
            pass
    # Fallback: nearest sentence containing first 3 fact words (no fabrication beyond source)
    try:
        words = str(fact_text or "").split()[:3]
        if words and chunk:
            low = chunk.lower()
            for sent, _, _ in _split_sentences(chunk):
                slow = sent.lower()
                if all(w.lower() in slow for w in words):
                    # Return first 10 words of that sentence as grounded span
                    s = " ".join(sent.split()[:10])[:100]
                    if s and s in chunk:
                        return False, s, "nearest-sentence-fallback"
    except Exception:
        pass
    return False, span, "invalid"
