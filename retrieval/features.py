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


def query_overlap(query, text):
    """Weighted Jaccard overlap using IDF-inspired weights."""
    q_tokens = set(tokenize(query))
    d_tokens = set(tokenize(text))
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


def semantic_similarity(query, text):
    """Cosine similarity between query and text embeddings."""
    q_emb = get_embedding(query)
    d_emb = get_embedding(text)
    if not q_emb or not d_emb:
        return 0.0
    import numpy as np
    q = np.array(q_emb, dtype=np.float32)
    d = np.array(d_emb, dtype=np.float32)
    return float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-8))


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
