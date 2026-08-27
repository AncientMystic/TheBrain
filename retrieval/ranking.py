"""
Ranking model for retrieval datapoints.
Uses a linear combination of features with learned weights.
For now, weights are set manually but can be replaced with a trained model.
"""
import config
from retrieval.features import compute_features
from typing import List, Dict


class LinearRanker:
    """Linear ranker with configurable weights."""
    def __init__(self, weights=None):
        self.weights = weights or getattr(config, 'RETRIEVAL_RANKING_WEIGHTS', None)
        if self.weights is None:
            self.weights = {
                'query_overlap': 0.25,
                'rare_term_boost': 0.1,
                'semantic_similarity': 0.2,
                'graph_proximity': 0.1,
                'entity_salience': 0.05,
                'doc_relevance': 0.1,
                'type_weight': 0.1,
                'confidence': 0.1,
            }
        total = sum(self.weights.values())
        if total == 0:
            total = 1
        self.weights = {k: v / total for k, v in self.weights.items()}

    def score(self, query, datapoint, query_entities, reranker=None):
        features = compute_features(query, datapoint, query_entities, reranker)
        keys = list(self.weights.keys())
        score = 0.0
        for i, key in enumerate(keys):
            if i < len(features):
                score += self.weights[key] * features[i]
        return score


class FallbackRanker:
    """Heuristic fallback similar to original but slightly improved."""
    def score(self, query, datapoint, query_entities, reranker=None):
        from core.text_utils import tokenize
        q_tokens = set(tokenize(query))
        text = datapoint.get('text', '') or ''
        d_tokens = set(tokenize(text))
        overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
        type_boost = 1.2 if datapoint.get('type') == 'fact' else 1.0
        conf = datapoint.get('confidence', 0.5)
        return overlap * type_boost + conf * 0.3


_ranker = None

def get_ranker():
    global _ranker
    if _ranker is None:
        if getattr(config, 'USE_LEARNED_RANKER', False):
            # Placeholder for loading a trained model
            _ranker = LinearRanker()
        else:
            _ranker = LinearRanker()
    return _ranker
