
"""
Multi-stage retrieval orchestrator.
Inclusive retrieval, dynamic hyperbolic scoring, no hard filters.
"""

import concurrent.futures
from typing import List, Dict, Any
import numpy as np
import config
from core import db
from core.embeddings import get_embeddings_batch
from retrieval.datapoint_retriever import retrieve_datapoints
from chat.retriever import retrieve_from_graph, fallback_to_chunks
from core.recoll_client import RecollClient
from retrieval.ranking import get_ranker
from graph.graph_queries import get_facts_by_keyword
from chat.query_analyzer import analyze_query
from core.metrics import inc_counter, Timer


def weighted_rrf(results_by_stage, weights, k=60):
    from collections import defaultdict
    scores = defaultdict(float)
    for stage, items in results_by_stage.items():
        w = weights.get(stage, 0.0)
        for rank, item in enumerate(items):
            item_id = item.get('id') if isinstance(item, dict) else str(item)
            scores[item_id] += w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def run_graph_retriever(query, analysis, top_k=None, anchor_entities=None):
    if anchor_entities:
        if isinstance(anchor_entities, list):
            for ent in anchor_entities:
                if isinstance(ent, str):
                    analysis["entities"].append({"text": ent})
    facts = retrieve_from_graph(analysis, top_k=top_k, max_depth=2)
    datapoints = []
    for f in facts:
        # Only skip malformed facts
        if not f.get('fact_text') or not f.get('doc_hash'):
            continue
        datapoints.append({
            'id': f"fact:{f.get('fact_id')}",
            'type': 'fact',
            'text': f.get('fact_text', ''),
            'doc_hash': f.get('doc_hash'),
            'doc_name': f.get('doc_name', 'unknown'),
            'confidence': f.get('confidence', 0.5),
            'verification_status': f.get('verification_status', 'unverified'),
            'chunk_id': f.get('chunk_id'),
            'source_span': f.get('source_span'),
        })
    return datapoints


def run_vector_retriever(query, top_k=200):
    chunks = fallback_to_chunks(query, top_k=top_k)
    datapoints = []
    for score, chunk_id, doc_hash, text in chunks:
        datapoints.append({
            'id': f"chunk:{chunk_id}",
            'type': 'chunk_ref',
            'text': text[:300],
            'doc_hash': doc_hash,
            'chunk_id': chunk_id,
            'confidence': 0.6,
            'verification_status': 'unverified',
        })
    return datapoints


def run_lexical_retriever(query, top_k=20):
    if not config.USE_RECOLL:
        return []
    try:
        client = RecollClient()
        results, _ = client.search(query, limit=top_k)
        datapoints = []
        for res in results:
            datapoints.append({
                'id': f"recoll:{res.get('path','')}",
                'type': 'chunk_ref',
                'text': res.get('snippet',''),
                'doc_hash': None,
                'confidence': 0.4,
                'verification_status': 'unverified',
            })
        return datapoints
    except Exception:
        return []


def run_gnn_retriever(query, top_k=10):
    if not getattr(config, "USE_GNN", False):
        return []
    # GNN is disabled or optional; we keep simple fallback
    return []


def run_topic_index_retriever(query, top_k=10):
    if not getattr(config, "USE_TOPIC_INDEX", True):
        return []
    try:
        from core.embeddings import get_embedding
        from core.streaming_topic_index import query_stream_topic_index as query_topic_index
        q_h = get_embedding(query, space='hyperbolic')
        if q_h is None:
            return []
        chunks = query_topic_index(q_h, top_clusters=5, chunks_per_cluster=3)
        datapoints = []
        for c in chunks:
            datapoints.append({
                'id': f"topic_chunk:{c['chunk_id']}",
                'type': 'chunk_ref',
                'text': c['text'],
                'doc_hash': c['doc_hash'],
                'chunk_id': c['chunk_id'],
                'confidence': 0.5,
                'verification_status': 'unverified',
            })
        return datapoints[:top_k]
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"  (Topic index retriever error: {e})")
        return []


class RetrievalOrchestrator:
    def __init__(self, stage_weights=None):
        self.stage_weights = stage_weights or getattr(config, 'RETRIEVAL_STAGE_WEIGHTS', None)
        if self.stage_weights is None:
            self.stage_weights = {
                'graph': 0.5,
                'vector': 0.3,
                'lexical': 0.2,
                'gnn': 0.0,
                'topic_index': 0.2,
                'hierarchical': 0.1,
            }
        if 'topic_index' not in self.stage_weights:
            self.stage_weights['topic_index'] = getattr(config, 'TOPIC_INDEX_STAGE_WEIGHT', 0.2)
        if 'hierarchical' not in self.stage_weights:
            self.stage_weights['hierarchical'] = 0.1
        total = sum(self.stage_weights.values())
        if total == 0:
            total = 1
        self.stage_weights = {k: v / total for k, v in self.stage_weights.items()}

    def _score_candidates(self, query, candidates, query_entities):
        from core.text_utils import tokenize
        from core.hyperbolic import hyperbolic_distance
        from core.embeddings import get_embeddings_batch
        import numpy as np

        q_embs = get_embeddings_batch([query], space='hyperbolic')
        q_emb = q_embs[0] if q_embs else None
        q_tokens = set(tokenize(query))

        conn_emb = db.db_connect("embeddings")
        cur_emb = conn_emb.cursor()
        conn_kf = db.db_connect("key_facts")
        cur_kf = conn_kf.cursor()

        for dp in candidates:
            text = dp.get('text', '') or ''
            d_tokens = set(tokenize(text))
            overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            rare_overlap = sum(1 for t in q_tokens if len(t) > 5 and t in d_tokens) / max(1, len([t for t in q_tokens if len(t) > 5]))
            lexical = 0.6 * overlap + 0.4 * rare_overlap

            emb = None
            fact_conf = dp.get('confidence', 0.5)
            if dp.get('type') == 'fact':
                fid = dp.get('id', '').split(':')[-1]
                if fid.isdigit():
                    cur_kf.execute("SELECT fact_embedding, confidence FROM key_facts WHERE fact_id=?", (int(fid),))
                    row = cur_kf.fetchone()
                    if row:
                        emb = np.frombuffer(row[0], dtype=np.float32) if row[0] else None
                        fact_conf = row[1] if row[1] is not None else fact_conf
            elif dp.get('type') == 'chunk_ref' and dp.get('chunk_id'):
                cur_emb.execute("SELECT embedding FROM chunk_embeddings WHERE chunk_id=?", (dp['chunk_id'],))
                row = cur_emb.fetchone()
                if row and row[0]:
                    emb = np.frombuffer(row[0], dtype=np.float32)
                fact_conf = 0.6

            hyper_sim = 0.0
            if q_emb is not None and emb is not None:
                dist = hyperbolic_distance(q_emb, emb)
                hyper_sim = 1.0 / (1.0 + dist)

            graph_prox = 0.0
            for ent in query_entities:
                if ent.lower() in text.lower():
                    graph_prox = 1.0
                    break

            conf = max(dp.get('confidence', 0.0), fact_conf)
            final = 0.45 * hyper_sim + 0.30 * lexical + 0.15 * graph_prox + 0.10 * conf
            dp['_final_score'] = final
            dp['_hyperbolic_sim'] = hyper_sim
            dp['_lexical_sim'] = lexical

        conn_emb.close()
        conn_kf.close()
        return candidates

    def retrieve(self, query, analysis, top_k=None, anchor_entities=None):
        inc_counter("retrieval_requests_total")
        with Timer("retrieval_duration_seconds"):
            stages = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                future_graph = executor.submit(run_graph_retriever, query, analysis, None, anchor_entities=anchor_entities)
                future_vector = executor.submit(run_vector_retriever, query, 200)
                future_lexical = executor.submit(run_lexical_retriever, query, 20)
                future_gnn = executor.submit(run_gnn_retriever, query, 10)
                future_topic = executor.submit(run_topic_index_retriever, query, 20)
                future_hier = executor.submit(retrieve_datapoints, query, max_nodes=200)

                stages['graph'] = future_graph.result()
                stages['vector'] = future_vector.result()
                stages['lexical'] = future_lexical.result()
                stages['gnn'] = future_gnn.result()
                stages['topic_index'] = future_topic.result()
                stages['hierarchical'] = future_hier.result()

            fused = weighted_rrf(stages, self.stage_weights)
            id_to_dp = {}
            for dps in stages.values():
                for dp in dps:
                    id_to_dp[dp['id']] = dp

            candidates = []
            for dp_id, score in fused:
                dp = id_to_dp.get(dp_id)
                if dp:
                    dp_copy = dp.copy()
                    dp_copy['_fused_score'] = score
                    candidates.append(dp_copy)

            query_entities = [ent.get('text', '') for ent in analysis.get('entities', []) if ent.get('text')]
            candidates = self._score_candidates(query, candidates, query_entities)

            candidates.sort(key=lambda x: x.get('_final_score', 0), reverse=True)
            if len(candidates) > 500:
                candidates = candidates[:500]
            return candidates
