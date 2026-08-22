"""Lightweight query intent detection."""
import re

def detect_intent(query):
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
