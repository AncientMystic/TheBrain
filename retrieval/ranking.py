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

    def batch_score(self, query, datapoints, query_entities, reranker=None):
        """
        Score multiple datapoints efficiently by batching embeddings.
        Returns list of scores aligned with datapoints.
        """
        from core.embeddings import get_embeddings_batch
        import numpy as np
        from retrieval.features import query_overlap, rare_term_boost, graph_proximity, \
            entity_salience, doc_relevance, datapoint_type_weight, confidence_feature

        # Collect all texts to embed: query + each datapoint text
        texts = [query] + [dp.get('text', '') or '' for dp in datapoints]
        embeddings = get_embeddings_batch(texts, batch_size=config.EMBEDDING_BATCH_SIZE)
        q_emb = embeddings[0] if embeddings else None
        dp_embs = embeddings[1:] if len(embeddings) > 1 else [None] * len(datapoints)

        scores = []
        for dp, dp_emb in zip(datapoints, dp_embs):
            text = dp.get('text', '') or ''
            features = []
            # 1. query_overlap
            features.append(query_overlap(query, text))
            # 2. rare_term_boost
            features.append(rare_term_boost(query, text))
            # 3. semantic_similarity (use precomputed embeddings)
            if q_emb is not None and dp_emb is not None:
                q = np.array(q_emb, dtype=np.float32)
                d = np.array(dp_emb, dtype=np.float32)
                sim = float(np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d) + 1e-8))
                features.append(sim)
            else:
                features.append(0.0)
            # 4. graph_proximity
            features.append(graph_proximity(dp, query_entities))
            # 5. entity_salience
            features.append(entity_salience(dp))
            # 6. doc_relevance
            features.append(doc_relevance(dp, reranker))
            # 7. type_weight
            features.append(datapoint_type_weight(dp.get('type')))
            # 8. confidence
            features.append(confidence_feature(dp))

            keys = list(self.weights.keys())
            score = 0.0
            for i, key in enumerate(keys):
                if i < len(features):
                    score += self.weights[key] * features[i]
            scores.append(score)
        return scores


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
