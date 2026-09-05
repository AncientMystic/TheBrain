
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
import logging
logger = logging.getLogger(__name__)


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
        logger.warning("Unexpected exception occurred", exc_info=True)
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
        logger.warning("Unexpected exception occurred", exc_info=True)
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
                'direct': 0.5,
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
        from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance_matrix
        from core.embeddings import get_embeddings_batch
        import numpy as np

        q_embs = get_embeddings_batch([query], space='hyperbolic')
        q_emb = q_embs[0] if q_embs else None
        if q_emb is not None:
            q_emb = ensure_hyperbolic(np.asarray(q_emb, dtype=np.float32), space='hyperbolic')
        q_tokens = set(tokenize(query))
        rare_q = [t for t in q_tokens if len(t) > 5]
        rare_denom = max(1, len(rare_q))

        # Batch-fetch stored embeddings + confidences (no N+1)
        fact_ids = []
        chunk_ids = []
        for dp in candidates:
            if dp.get('type') == 'fact':
                fid = dp.get('id', '').split(':')[-1]
                if fid.isdigit():
                    fact_ids.append(int(fid))
            elif dp.get('type') == 'chunk_ref' and dp.get('chunk_id'):
                chunk_ids.append(dp['chunk_id'])
        fact_map = {}
        fact_conf_map = {}
        chunk_map = {}
        try:
            import config as _cfg_o
            _exp_dim = int(getattr(_cfg_o, "EMBEDDING_DIM", 1024))
        except Exception:
            _exp_dim = 1024
        conn_emb = db.db_connect("embeddings")
        conn_kf = db.db_connect("key_facts")
        try:
            cur_kf = conn_kf.cursor()
            for s in range(0, len(fact_ids), 400):
                ch = fact_ids[s:s+400]
                if not ch:
                    continue
                ph = ",".join("?" for _ in ch)
                cur_kf.execute(f"SELECT fact_id, fact_embedding, confidence FROM key_facts WHERE fact_id IN ({ph})", ch)
                for r in cur_kf.fetchall():
                    if r[1] is not None:
                        try:
                            _arr = np.frombuffer(r[1], dtype=np.float32)
                            if len(_arr) != _exp_dim:
                                continue
                            fact_map[r[0]] = _arr.copy()
                        except Exception:
                            continue
                    if r[2] is not None:
                        fact_conf_map[r[0]] = r[2]
            cur_emb = conn_emb.cursor()
            for s in range(0, len(chunk_ids), 400):
                ch = chunk_ids[s:s+400]
                if not ch:
                    continue
                ph = ",".join("?" for _ in ch)
                cur_emb.execute(f"SELECT chunk_id, embedding FROM chunk_embeddings WHERE chunk_id IN ({ph})", ch)
                for r in cur_emb.fetchall():
                    if r[1] is not None:
                        try:
                            _arr = np.frombuffer(r[1], dtype=np.float32)
                            if len(_arr) != _exp_dim:
                                continue
                            chunk_map[r[0]] = _arr.copy()
                        except Exception:
                            continue
        finally:
            try:
                conn_emb.close()
            except Exception:
                pass
            try:
                conn_kf.close()
            except Exception:
                pass
        # Vectorized sims for present embs
        ordered = []
        present_idx = []
        for di, dp in enumerate(candidates):
            e = None
            if dp.get('type') == 'fact':
                fid = dp.get('id', '').split(':')[-1]
                if fid.isdigit():
                    e = fact_map.get(int(fid))
            elif dp.get('type') == 'chunk_ref' and dp.get('chunk_id'):
                e = chunk_map.get(dp['chunk_id'])
            if e is not None:
                ordered.append(ensure_hyperbolic(e, space='hyperbolic'))
                present_idx.append(di)
        sim_map = {}
        if q_emb is not None and ordered:
            pmat = np.stack(ordered)
            dists = hyperbolic_distance_matrix(q_emb[None, :], pmat)[0]
            for idx, d in zip(present_idx, dists):
                sim_map[idx] = float(1.0 / (1.0 + float(d)))

        for di, dp in enumerate(candidates):
            text = dp.get('text', '') or ''
            d_tokens = set(tokenize(text))
            overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
            rare_overlap = sum(1 for t in rare_q if t in d_tokens) / rare_denom
            lexical = 0.6 * overlap + 0.4 * rare_overlap

            fact_conf = dp.get('confidence', 0.5)
            if dp.get('type') == 'fact':
                fid = dp.get('id', '').split(':')[-1]
                if fid.isdigit() and int(fid) in fact_conf_map:
                    fact_conf = fact_conf_map[int(fid)]
            elif dp.get('type') == 'chunk_ref' and dp.get('chunk_id'):
                fact_conf = 0.6

            hyper_sim = sim_map.get(di, 0.0)

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

        return candidates

    def retrieve(self, query, analysis, top_k=None, anchor_entities=None):
        inc_counter("retrieval_requests_total")
        with Timer("retrieval_duration_seconds"):
            stages = {}
            direct_dps = run_direct_document_retriever(query)
            if direct_dps:
                stages['direct'] = direct_dps
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


def run_direct_document_retriever(query, top_k=5):
    """Retrieve documents whose filename/title matches a direct numeric/episode reference."""
    import re
    from core import db

    patterns = [
        r'episode\s*(\d+)',
        r'\b(\d{2,3})\b',
    ]
    doc_matches = []
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            num = m.group(1)
            conn = db.db_connect("index")
            cur = conn.cursor()
            cur.execute("SELECT file_hash, filename, title FROM documents WHERE filename LIKE ? OR title LIKE ? LIMIT ?",
                        (f'%{num}%', f'%{num}%', top_k))
            rows = cur.fetchall()
            conn.close()
            doc_matches.extend(rows)
            break

    datapoints = []
    seen_hashes = set()
    for d in doc_matches:
        doc_hash = d['file_hash']
        if doc_hash in seen_hashes:
            continue
        seen_hashes.add(doc_hash)
        conn = db.db_connect("key_facts")
        cur = conn.cursor()
        cur.execute("SELECT fact_id, doc_hash, doc_name, fact_text, canonical_value, source_span, confidence FROM key_facts WHERE doc_hash=? ORDER BY confidence DESC LIMIT 200", (doc_hash,))
        facts = cur.fetchall()
        conn.close()
        for f in facts:
            datapoints.append({
                'id': f"direct_fact:{f['fact_id']}",
                'type': 'fact',
                'text': f['fact_text'],
                'doc_hash': f['doc_hash'],
                'doc_name': f['doc_name'],
                'confidence': f['confidence'],
                'source_span': f['source_span'],
            })
        conn = db.db_connect("index")
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, doc_hash, chunk_text FROM document_chunks WHERE doc_hash=? ORDER BY chunk_index LIMIT 20", (doc_hash,))
        chunks = cur.fetchall()
        conn.close()
        for c in chunks:
            datapoints.append({
                'id': f"direct_chunk:{c['chunk_id']}",
                'type': 'chunk_ref',
                'text': c['chunk_text'][:500],
                'doc_hash': c['doc_hash'],
                'chunk_id': c['chunk_id'],
                'confidence': 0.8,
            })
    return datapoints
