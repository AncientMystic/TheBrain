
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

    def score(self, query, datapoint, query_entities, reranker=None):
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
        from core.text_utils import tokenize as _tok
        from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance_matrix
        import numpy as np

        # Embed query only once (batched) + tokenize once (not per-datapoint)
        query_embs = get_embeddings_batch([query], space='hyperbolic')
        q_emb = query_embs[0] if query_embs else None
        if q_emb is not None:
            q_emb = ensure_hyperbolic(np.asarray(q_emb, dtype=np.float32), space='hyperbolic')
        try:
            qtok = set(_tok(query))
        except Exception:
            qtok = set(query.lower().split())

        # Batch-fetch stored embeddings (avoid N+1 SELECTs, 400-chunk IN)
        fact_ids = []
        chunk_ids = []
        for dp in datapoints:
            if dp.get('type') == 'fact':
                fid = dp.get('id', '').split(':')[-1]
                if fid.isdigit():
                    fact_ids.append(int(fid))
            elif dp.get('type') == 'chunk_ref' and dp.get('chunk_id'):
                chunk_ids.append(dp['chunk_id'])
        fact_map = {}
        chunk_map = {}
        conn_emb = db.db_connect("embeddings")
        conn_kf = db.db_connect("key_facts")
        try:
            cur_kf = conn_kf.cursor()
            for s in range(0, len(fact_ids), 400):
                ch = fact_ids[s:s+400]
                if not ch:
                    continue
                ph = ",".join("?" for _ in ch)
                cur_kf.execute(f"SELECT fact_id, fact_embedding FROM key_facts WHERE fact_id IN ({ph})", ch)
                for r in cur_kf.fetchall():
                    if r[1] is not None:
                        fact_map[r[0]] = np.frombuffer(r[1], dtype=np.float32).copy()
            cur_emb = conn_emb.cursor()
            for s in range(0, len(chunk_ids), 400):
                ch = chunk_ids[s:s+400]
                if not ch:
                    continue
                ph = ",".join("?" for _ in ch)
                cur_emb.execute(f"SELECT chunk_id, embedding FROM chunk_embeddings WHERE chunk_id IN ({ph})", ch)
                for r in cur_emb.fetchall():
                    if r[1] is not None:
                        chunk_map[r[0]] = np.frombuffer(r[1], dtype=np.float32).copy()
        finally:
            try:
                conn_emb.close()
            except Exception:
                pass
            try:
                conn_kf.close()
            except Exception:
                pass

        # Vectorized semantic sims: single distance_matrix call for all present embs
        ordered_vecs = []
        present_idx = []
        for di, dp in enumerate(datapoints):
            e = None
            if dp.get('type') == 'fact':
                fid = dp.get('id', '').split(':')[-1]
                if fid.isdigit():
                    e = fact_map.get(int(fid))
            elif dp.get('type') == 'chunk_ref' and dp.get('chunk_id'):
                e = chunk_map.get(dp['chunk_id'])
            if e is not None:
                ordered_vecs.append(ensure_hyperbolic(e, space='hyperbolic'))
                present_idx.append(di)
        pmap = {}
        if q_emb is not None and ordered_vecs:
            pmat = np.stack(ordered_vecs)
            dists = hyperbolic_distance_matrix(q_emb[None, :], pmat)[0]
            for idx, d in zip(present_idx, dists):
                pmap[idx] = float(1.0 / (1.0 + float(d)))

        scores = []
        for di, dp in enumerate(datapoints):
            text = dp.get('text', '') or ''
            features = []

            # 1. query_overlap (reuse qtok)
            try:
                dtok = set(_tok(text))
            except Exception:
                dtok = set(text.lower().split())
            features.append(len(qtok & dtok) / max(1, len(qtok)))
            # 2. rare_term_boost
            features.append(rare_term_boost(query, text))
            # 3. semantic_similarity from pre-fetched map
            sim = pmap.get(di, 0.0)
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
