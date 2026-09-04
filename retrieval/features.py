"""
Feature extraction for retrieval ranking.
Each function takes a query and a Datapoint-like dict and returns a float feature.
"""
import math
import re
from collections import Counter
from core.embeddings import get_embedding
from core.text_utils import tokenize
import config


def idf_scores(corpus_size=10000):
    """Return a dummy IDF mapping; in production, compute from actual corpus."""
    def idf(token):
        return math.log(1 + corpus_size / (1 + len(token) * 10))
    return idf


def _jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def query_overlap_tokens(qtok, text):
    """Weighted Jaccard overlap given pre-tokenized query (avoids retokenize per datapoint)."""
    d_tokens = set(tokenize(text))
    if not qtok:
        return 0.0
    idf = idf_scores()
    weighted_intersection = sum(idf(t) for t in qtok & d_tokens)
    weighted_union = sum(idf(t) for t in qtok | d_tokens)
    if weighted_union == 0:
        return 0.0
    return weighted_intersection / weighted_union


def query_overlap(query, text):
    """Weighted Jaccard overlap using IDF-inspired weights."""
    q_tokens = set(tokenize(query))
    return query_overlap_tokens(q_tokens, text)
    if not q_tokens:
        return 0.0
    idf = idf_scores()
    weighted_intersection = sum(idf(t) for t in q_tokens & d_tokens)
    weighted_union = sum(idf(t) for t in q_tokens | d_tokens)
    if weighted_union == 0:
        return 0.0
    return weighted_intersection / weighted_union


def rare_term_boost(query, text):
    """Count of rare query terms (long tokens) present in text."""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    rare = [t for t in q_tokens if len(t) > 5]
    if not rare:
        return 0.0
    d_tokens = set(tokenize(text))
    return sum(1 for t in rare if t in d_tokens) / len(rare)


def semantic_similarity(query, text, q_emb=None, emb_map=None):
    """Hyperbolic similarity between query and text embeddings.

    Accepts pre-fetched q_emb / emb_map to avoid per-call HTTP (single batch upstream).
    Falls back to singular get_embedding only for backward compat.
    """
    from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance
    import numpy as _np
    if q_emb is None:
        q_emb = get_embedding(query, space='hyperbolic')
    if emb_map is not None:
        d_emb = emb_map.get(text)
    else:
        d_emb = get_embedding(text, space='hyperbolic')
    if q_emb is None or d_emb is None:
        return 0.0
    try:
        qh = ensure_hyperbolic(_np.asarray(q_emb, dtype=_np.float32), space='hyperbolic')
        dh = ensure_hyperbolic(_np.asarray(d_emb, dtype=_np.float32), space='hyperbolic')
        dist = float(hyperbolic_distance(qh, dh))
    except Exception:
        return 0.0
    return 1.0 / (1.0 + dist)


def semantic_similarities_batched(query, texts, q_emb=None):
    """Vectorized sims for many texts: single query embed + single distance_matrix."""
    from core.embeddings import get_embeddings_dict
    from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance_matrix
    import numpy as _np
    if q_emb is None:
        q_emb = get_embedding(query, space='hyperbolic')
    if q_emb is None:
        return [0.0] * len(texts)
    qh = ensure_hyperbolic(_np.asarray(q_emb, dtype=_np.float32), space='hyperbolic')[None, :]
    emb_map = get_embeddings_dict([t for t in texts if t], space='hyperbolic')
    vecs = []
    valid = []
    for t in texts:
        e = emb_map.get(t)
        if e is None:
            valid.append(False)
        else:
            vecs.append(ensure_hyperbolic(e, space='hyperbolic'))
            valid.append(True)
    if not vecs:
        return [0.0] * len(texts)
    import numpy as _np2
    pmat = _np2.stack(vecs)
    dists = hyperbolic_distance_matrix(qh, pmat)[0]
    out = []
    di = 0
    for v in valid:
        if not v:
            out.append(0.0)
        else:
            out.append(float(1.0 / (1.0 + float(dists[di]))))
            di += 1
    return out

def graph_proximity(datapoint, query_entities, max_depth=2):
    """Inverse distance from query entities in external graph."""
    from graph.graph_queries import get_global_node_edges
    from core import db
    overlap = 0
    text = (datapoint.get('text') or '').lower()
    for ent in query_entities:
        if ent.lower() in text:
            overlap += 1
    if overlap == 0:
        return 0.0
    return min(1.0, overlap / len(query_entities))


def entity_salience(datapoint):
    """Importance of entities mentioned in datapoint, based on node degree."""
    return 0.5


def doc_relevance(datapoint, reranker):
    """Use reranker score of the parent document/summary if available."""
    return datapoint.get('_doc_relevance', 0.0)


def datapoint_type_weight(dp_type):
    """Prior importance of each type."""
    weights = {
        'fact': 0.9,
        'chunk_ref': 0.6,
        'summary': 0.4,
        'document': 0.2,
        'entity': 0.8,
    }
    return weights.get(dp_type, 0.5)


def confidence_feature(datapoint):
    return datapoint.get('confidence', 0.0)


def compute_features(query, datapoint, query_entities, reranker=None):
    """
    Compute a feature vector for a datapoint given the query.
    Returns list of floats in a fixed order.
    """
    text = datapoint.get('text', '') or ''
    features = []
    features.append(query_overlap(query, text))
    features.append(rare_term_boost(query, text))
    features.append(semantic_similarity(query, text))
    features.append(graph_proximity(datapoint, query_entities))
    features.append(entity_salience(datapoint))
    features.append(doc_relevance(datapoint, reranker))
    features.append(datapoint_type_weight(datapoint.get('type')))
    features.append(confidence_feature(datapoint))
    return features
