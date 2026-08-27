"""
Rank extracted facts/entities against document title or context.
Uses local reranker if available; falls back to simple confidence.
"""

import config


def rank_extracted_items(doc_title, items, text_field="fact_text"):
    """
    Score and return items sorted by relevance to doc_title.

    If reranker is available, use it. Otherwise return items unchanged.
    """
    if not items:
        return items

    try:
        from core.reranker import get_reranker
        reranker = get_reranker()
        if reranker.available:
            texts = [item.get(text_field, "") for item in items if isinstance(item, dict)]
            if texts:
                scores = reranker.score(doc_title, texts)
                for item, score in zip(items, scores):
                    item["_ingest_score"] = score
                items.sort(key=lambda x: x.get("_ingest_score", 0), reverse=True)
                return items
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"    (Ingest reranker error: {e})")

    return items
