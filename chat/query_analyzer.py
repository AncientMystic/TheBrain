"""Backward-compat shim: query analysis now lives in core/ (no chat dependency).

Breaks the former chat<->retrieval import fragility: lower layers import from
core, chat re-exports for existing callers.
"""
from core.query_analyzer import analyze_query, extract_topic_terms

__all__ = ["analyze_query", "extract_topic_terms"]
