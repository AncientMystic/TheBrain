
import re
import logging
logger = logging.getLogger(__name__)

def _safe_str(value, max_len=200):
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception as e:
            logger.warning("Unexpected exception occurred", exc_info=True)
            return ""
    return value[:max_len]

def _coerce_float(value, default=0.0):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception as e:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return default

def validate_fact(item):
    if not isinstance(item, dict):
        return None
    fact_text = _safe_str(item.get("fact_text"), 200)
    if not fact_text:
        return None
    return {
        "fact_type": _safe_str(item.get("fact_type"), 80),
        "fact_text": fact_text,
        "canonical_value": _safe_str(item.get("canonical_value"), 80),
        "source_span": _safe_str(item.get("source_span"), 200),
        "confidence": _coerce_float(item.get("confidence"), 0.5),
        # preserve extra metadata if needed
    }

def validate_entity(item):
    if not isinstance(item, dict):
        return None
    entity_name = _safe_str(item.get("entity_name"), 150)
    if not entity_name:
        return None
    return {
        "entity_type": _safe_str(item.get("entity_type"), 40),
        "entity_name": entity_name,
        "normalized_name": _safe_str(item.get("normalized_name"), 150),
        "source_span": _safe_str(item.get("source_span"), 200),
        "confidence": _coerce_float(item.get("confidence"), 0.5),
    }

def validate_relationship(item):
    if not isinstance(item, dict):
        return None
    src = _safe_str(item.get("source_node"), 150)
    tgt = _safe_str(item.get("target_node"), 150)
    if not src or not tgt:
        return None
    return {
        "source_node": src,
        "target_node": tgt,
        "relation_type": _safe_str(item.get("relation_type"), 80),
        "evidence_span": _safe_str(item.get("evidence_span"), 200),
        "confidence": _coerce_float(item.get("confidence"), 0.5),
    }

def validate_and_coerce(category, item):
    if category == "facts":
        return validate_fact(item)
    elif category == "entities":
        return validate_entity(item)
    elif category == "relationships":
        return validate_relationship(item)
    # For other categories, we can fallback to a simple coercion
    # but they are less critical; use a generic cleaner
    return item  # placeholder; will be overridden by cleaners in pipeline
