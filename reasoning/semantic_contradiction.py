"""
Semantic contradiction detection using embeddings.
"""
import numpy as np
from core.embeddings import get_embeddings_batch
from core.fact_normalizer import normalize_name
from core import db
import config


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if getattr(config, "USE_HYPERBOLIC_RETRIEVAL", False):
        from core.hyperbolic import exp_map, hyperbolic_distance
        dist = hyperbolic_distance(exp_map(a), exp_map(b))
        return float(1.0 / (1.0 + dist))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def detect_semantic_contradictions(facts, standards, threshold=0.85):
    """
    Compare unverified facts against standards using embeddings.
    Return list of contradiction dicts.
    """
    if not facts or not standards:
        return []

    fact_texts = [f.get("fact_text", "") for f in facts]
    standard_texts = [s.get("statement", "") for s in standards]

    fact_embs = get_embeddings_batch(fact_texts, batch_size=config.EMBEDDING_BATCH_SIZE)
    standard_embs = get_embeddings_batch(standard_texts, batch_size=config.EMBEDDING_BATCH_SIZE)

    contradictions = []
    for i, f_emb in enumerate(fact_embs):
        if f_emb is None:
            continue
        for j, s_emb in enumerate(standard_embs):
            if s_emb is None:
                continue
            sim = cosine_similarity(f_emb, s_emb)
            # Contradiction if high similarity but opposite truth value (negation differs)
            f_neg = facts[i].get("negation", 0)
            s_neg = standards[j].get("negation", 0)
            if sim > threshold and f_neg != s_neg:
                contradictions.append({
                    "fact_id": facts[i].get("fact_id"),
                    "standard_id": standards[j].get("id"),
                    "similarity": sim,
                    "reason": "high similarity with opposite negation",
                })
    return contradictions
