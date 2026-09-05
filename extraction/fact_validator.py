from core.embeddings import get_embedding
from core.text_utils import normalise_text
import numpy as np
import logging
logger = logging.getLogger(__name__)


def hyperbolic_similarity(a, b):
    from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance
    ah = ensure_hyperbolic(a, space='hyperbolic')
    bh = ensure_hyperbolic(b, space='hyperbolic')
    try:
        d = float(hyperbolic_distance(ah, bh))
    except Exception:
        return 0.0
    return 1.0 / (1.0 + d)


def cosine_similarity(a, b):
    # Backward-compat alias: geometry-correct via hyperbolic distance.
    return hyperbolic_similarity(a, b)


def validate_source_span(source_span: str, full_text: str) -> bool:
    """Check if source_span appears in full_text (exact, then fuzzy)."""
    if not source_span:
        return False
    if source_span in (full_text or ""):
        return True
    # Remove '...' and spaces for simple containment
    normalized_span = normalise_text(source_span.replace('...', ' ')).lower()
    normalized_full = normalise_text(full_text).lower()
    return normalized_span in normalized_full


def validate_source_span_detailed(fact_text: str, source_span: str, chunk_text: str,
                                  start_char=None, end_char=None):
    """Detailed check via span_validation helper (exact/offsets/fallback + reason)."""
    try:
        from core.span_validation import validate_span
        return validate_span(fact_text, source_span, chunk_text, start_char, end_char)
    except Exception:
        return validate_source_span(source_span, chunk_text or ""), source_span, "legacy"


def validate_embedding_similarity(entity_text: str, chunk_text: str, threshold: float = 0.35) -> bool:
    """Check semantic similarity between entity and source chunk."""
    emb1 = get_embedding(entity_text)
    emb2 = get_embedding(chunk_text)
    if emb1 is None or emb2 is None:
        return False
    sim = cosine_similarity(emb1, emb2)
    return sim >= threshold


def validate_item(item: dict, chunk_text: str, full_text: str) -> bool:
    """
    Perform validation on a single extracted item (span grounded + optional sem check).
    Corrects correctable spans in place when fallback is grounded in source.
    Returns True if item passes, else False.
    """
    fact_text = item.get("fact_text") or item.get("entity_name") or ""
    source_span = item.get("source_span", "")
    valid, corrected, _reason = validate_source_span_detailed(
        fact_text, source_span, chunk_text or full_text or "",
        item.get("start_char"), item.get("end_char"))
    if valid:
        return True
    # Accept grounded fallback correction (still in source, verifier will weight it)
    if corrected and corrected in (chunk_text or full_text or ""):
        try:
            item["source_span"] = corrected
            item["span_corrected"] = True
        except Exception:
            pass
        return True
    return validate_source_span(source_span, full_text)
