"""
Multi-stage retrieval orchestrator.
Runs graph, vector, lexical, and optionally GNN retrievers in parallel,
then fuses results using Weighted Reciprocal Rank Fusion (WRRF).
"""
import concurrent.futures
from typing import List, Dict, Any
import config
from retrieval.datapoint_retriever import retrieve_datapoints
from chat.retriever import retrieve_from_graph, fallback_to_chunks
from core.recoll_client import RecollClient
from retrieval.ranking import get_ranker
from core.metrics import inc_counter, observe_histogram, Timer


def weighted_rrf(results_by_stage: Dict[str, List[Any]], weights: Dict[str, float], k: int = 60):
    from collections import defaultdict
    scores = defaultdict(float)
    for stage, items in results_by_stage.items():
        w = weights.get(stage, 0.0)
        for rank, item in enumerate(items):
            item_id = item.get('id') if isinstance(item, dict) else str(item)
            scores[item_id] += w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def run_graph_retriever(query, analysis, top_k=50):
    facts = retrieve_from_graph(analysis, top_k=top_k, max_depth=2)
    datapoints = []
    for f in facts:
        datapoints.append({
            'id': f"fact:{f.get('fact_id')}",
            'type': 'fact',
            'text': f.get('fact_text', ''),
            'doc_hash': f.get('doc_hash'),
            'confidence': f.get('confidence', 0.5),
            'chunk_id': f.get('chunk_id'),
            'source_span': f.get('source_span'),
        })
    return datapoints


def run_vector_retriever(query, top_k=20):
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
            path = res.get('path', '')
            snippet = res.get('snippet', '')
            datapoints.append({
                'id': f"recoll:{path}",
                'type': 'chunk_ref',
                'text': snippet,
                'doc_hash': None,
                'confidence': 0.4,
            })
        return datapoints
    except Exception:
        return []


def run_gnn_retriever(query, top_k=10):
    """Retrieve datapoints using GNN node embeddings."""
    if not getattr(config, "USE_GNN", False):
        return []
    try:
        from graph.gnn_sage import get_gnn_embeddings
        import numpy as np
        from core.embeddings import get_embedding

        node_embs = get_gnn_embeddings()
        if node_embs is None:
            return []
        q_emb = get_embedding(query)
        if not q_emb:
            return []
        q_vec = np.array(q_emb, dtype=np.float32)
        # Normalize
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
        q_vec = q_vec / q_norm

        # Compute cosine similarities
        node_norms = np.linalg.norm(node_embs, axis=1, keepdims=True)
        node_norms[node_norms == 0] = 1e-8
        normalized_nodes = node_embs / node_norms
        sims = np.dot(normalized_nodes, q_vec)

        top_indices = np.argsort(sims)[-top_k:][::-1]

        # Map node embeddings to node IDs and then to facts via entity_fact_index
        from core import db
        conn = db.db_connect("external_graph")
        cur = conn.cursor()
        datapoints = []
        for idx in top_indices:
            # Need node id: we stored node_ids.npy
            try:
                node_ids = np.load(Path(config.GNN_MODEL_DIR) / "node_ids.npy")
                node_id = node_ids[idx]
            except Exception:
                node_id = idx  # fallback
            # Get node name
            cur.execute("SELECT canonical_name FROM global_nodes WHERE global_node_id=?", (node_id,))
            row = cur.fetchone()
            if not row:
                continue
            node_name = row["canonical_name"]
            # Retrieve facts associated with this entity via entity_fact_index
            cur2 = conn.cursor()
            cur2.execute("""
                SELECT f.fact_id, f.doc_hash, f.fact_text, f.canonical_value, f.source_span, f.confidence
                FROM key_facts f
                JOIN entity_fact_index efi ON f.fact_id = efi.fact_id
                WHERE efi.normalized_name = ?
                LIMIT 5
            """, (node_name.lower(),))
            facts = cur2.fetchall()
            for f in facts:
                datapoints.append({
                    'id': f"gnn_fact:{f['fact_id']}",
                    'type': 'fact',
                    'text': f['fact_text'],
                    'doc_hash': f['doc_hash'],
                    'confidence': f['confidence'],
                    'source_span': f['source_span'],
                })
        conn.close()
        return datapoints
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"GNN retriever error: {e}")
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
            }
        total = sum(self.stage_weights.values())
        if total == 0:
            total = 1
        self.stage_weights = {k: v / total for k, v in self.stage_weights.items()}

    def retrieve(self, query, analysis, top_k=30):
        inc_counter("retrieval_requests_total")
        with Timer("retrieval_duration_seconds"):
            stages = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_graph = executor.submit(run_graph_retriever, query, analysis, 50)
                future_vector = executor.submit(run_vector_retriever, query, 20)
                future_lexical = executor.submit(run_lexical_retriever, query, 20)
                future_gnn = executor.submit(run_gnn_retriever, query, 10)

                stages['graph'] = future_graph.result()
                stages['vector'] = future_vector.result()
                stages['lexical'] = future_lexical.result()
                stages['gnn'] = future_gnn.result()

        # Additional hierarchical datapoint retrieval
        try:
            hierarchical_dps = retrieve_datapoints(query, max_nodes=50)
            stages['hierarchical'] = hierarchical_dps
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"Hierarchical retriever failed: {e}")

        fused = weighted_rrf(stages, self.stage_weights)
        id_to_dp = {}
        for stage, dps in stages.items():
            for dp in dps:
                id_to_dp[dp['id']] = dp

        final_dps = []
        for dp_id, score in fused[:top_k]:
            dp = id_to_dp.get(dp_id)
            if dp:
                dp = dp.copy()
                dp['_fused_score'] = score
                final_dps.append(dp)
        return final_dps
