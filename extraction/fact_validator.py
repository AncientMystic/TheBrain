from core.embeddings import get_embedding
from core.text_utils import normalise_text
import numpy as np


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def validate_source_span(source_span: str, full_text: str) -> bool:
    """Check if source_span appears in full_text (fuzzy)."""
    if not source_span:
        return False
    # Remove '...' and spaces for simple containment
    normalized_span = normalise_text(source_span.replace('...', ' ')).lower()
    normalized_full = normalise_text(full_text).lower()
    return normalized_span in normalized_full


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
    Perform basic validation on a single extracted item.
    Returns True if item passes, else False.
    """
    source_span = item.get("source_span", "")
    if not validate_source_span(source_span, full_text):
        return False
    # Optional embedding check if available
    # if "entity_name" in item or "fact_text" in item:
    #     text_to_check = item.get("entity_name") or item.get("fact_text")
    #     if text_to_check and not validate_embedding_similarity(text_to_check, chunk_text):
    #         return False
    return True