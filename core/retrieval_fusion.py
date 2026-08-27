"""
Weighted Reciprocal Rank Fusion for multi-stage retrieval.
"""
import config


def weighted_rrf(results_by_stage, weights=None, k=60):
    """
    Combine multiple retrieval stage results using Weighted RRF.

    Args:
        results_by_stage: dict mapping stage name -> list of items with 'id'
        weights: dict mapping stage name -> weight (default from config)
        k: constant to reduce impact of high ranks

    Returns:
        list of (id, score) sorted descending
    """
    if weights is None:
        weights = getattr(config, "RETRIEVAL_STAGE_WEIGHTS", {})

    scores = {}
    for stage_name, stage_results in results_by_stage.items():
        w = weights.get(stage_name, 0.0)
        for rank, item in enumerate(stage_results):
            item_id = item.get("id") or item.get("fact_id") or item.get("chunk_id") or str(item)
            scores[item_id] = scores.get(item_id, 0.0) + w / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
