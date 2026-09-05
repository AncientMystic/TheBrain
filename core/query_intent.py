"""Lightweight query intent detection."""
import re
import logging
logger = logging.getLogger(__name__)

def detect_intent(query):
    # Use neural classifier if available
    try:
        from core.intent_classifier import get_intent_classifier
        classifier = get_intent_classifier()
        if classifier.available:
            labels = ["factual", "summary", "detail", "comparative", "causal", "temporal", "general"]
            label = classifier.classify(query, labels)
            if label in labels:
                return label
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        pass
    # Fallback to rule-based
    q = query.lower()
    if any(w in q for w in ["compare", "versus", "vs", "difference"]):
        return "comparative"
    if any(w in q for w in ["summarize", "summary", "sum up"]):
        return "summary"
    if any(w in q for w in ["why", "cause", "reason"]):
        return "causal"
    if any(w in q for w in ["before", "after", "timeline", "when did"]):
        return "temporal"
    if any(w in q for w in ["what", "who", "where", "which", "how many"]):
        return "factual"
    return "general"
