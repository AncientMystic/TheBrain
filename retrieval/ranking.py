
"""
Ranking model for retrieval datapoints using stored hyperbolic embeddings.
"""

import numpy as np
import config
from retrieval.features import (
    query_overlap,
    rare_term_boost,
    semantic_similarity,
    graph_proximity,
    entity_salience,
    doc_relevance,
    datapoint_type_weight,
    confidence_feature,
)
from core import db
from core.embeddings import get_embeddings_batch
from core.hyperbolic import hyperbolic_distance


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

    def score(self, query, datapoint, _query_entities, _reranker=None):
        features = self._compute_features(query, datapoint, query_entities, reranker)
        keys = list(self.weights.keys())
        score = 0.0
        for i, key in enumerate(keys):
            if i < len(features):
                score += self.weights[key] * features[i]
        return score

    def batch_score(self, query, datapoints, query_entities, reranker=None):
        """Score multiple datapoints efficiently using stored hyperbolic embeddings."""
        from core.embeddings import get_embeddings_batch
        import numpy as np

        # Embed query only once (batched)
        query_embs = get_embeddings_batch([query], space='hyperbolic')
        q_emb = query_embs[0] if query_embs else None

        conn_emb = db.db_connect("embeddings")
        cur_emb = conn_emb.cursor()
        conn_kf = db.db_connect("key_facts")
        cur_kf = conn_kf.cursor()

        scores = []
        for dp in datapoints:
            text = dp.get('text', '') or ''
            features = []

            # 1. query_overlap
            features.append(query_overlap(query, text))
            # 2. rare_term_boost
            features.append(rare_term_boost(query, text))
            # 3. semantic_similarity from stored embedding
            emb = None
            if dp.get('type') == 'fact':
                fid = dp.get('id', '').split(':')[-1]
                if fid.isdigit():
                    cur_kf.execute("SELECT fact_embedding FROM key_facts WHERE fact_id=?", (int(fid),))
                    row = cur_kf.fetchone()
                    if row and row[0]:
                        emb = np.frombuffer(row[0], dtype=np.float32)
            elif dp.get('type') == 'chunk_ref' and dp.get('chunk_id'):
                cur_emb.execute("SELECT embedding FROM chunk_embeddings WHERE chunk_id=?", (dp['chunk_id'],))
                row = cur_emb.fetchone()
                if row and row[0]:
                    emb = np.frombuffer(row[0], dtype=np.float32)

            if q_emb is not None and emb is not None:
                dist = hyperbolic_distance(q_emb, emb)
                sim = 1.0 / (1.0 + dist)
            else:
                sim = 0.0
            features.append(sim)
            # 4. graph_proximity
            features.append(graph_proximity(dp, query_entities))
            # 5. entity_salience
            features.append(entity_salience(dp))
            # 6. doc_relevance
            features.append(doc_relevance(dp, reranker))
            # 7. datapoint_type_weight
            features.append(datapoint_type_weight(dp.get('type')))
            # 8. confidence
            features.append(confidence_feature(dp))

            keys = list(self.weights.keys())
            score = 0.0
            for i, key in enumerate(keys):
                if i < len(features):
                    score += self.weights[key] * features[i]
            scores.append(score)

        conn_emb.close()
        conn_kf.close()
        return scores

    def _compute_features(self, query, datapoint, query_entities, reranker):
        # Wrapper for single scoring using batch_score
        return self.batch_score(query, [datapoint], query_entities, reranker)[0]


class FallbackRanker:
    """Heuristic fallback ranker."""
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
        _ranker = LinearRanker()
    return _ranker
