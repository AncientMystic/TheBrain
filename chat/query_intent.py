"""Backward-compat shim: query intent now lives in core/ (no chat dependency).

Breaks the former chat<->retrieval import fragility: lower layers import from
core, chat re-exports for existing callers.
"""
from core.query_intent import detect_intent

__all__ = ["detect_intent"]
