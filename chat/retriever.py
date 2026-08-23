import numpy as np
import json
import time
from core import db
from core.embeddings import get_embedding
from graph.graph_queries import get_related_keywords, get_facts_by_keyword, get_global_node_edges
from graph.expansion import expand_facts_via_multi_hop
from logic.retrieve import retrieve_logic_modules
from memory.retrieve import retrieve_memories
import config

_fact_cache = {}
_fact_cache_ttl = {}

def _cached_get_facts(key, ttl=300):
    now = time.time()
    if key in _fact_cache and now - _fact_cache_ttl.get(key, 0) < ttl:
        return _fact_cache[key]
    return None

def _cache_facts(key, facts, ttl=300):
    _fact_cache[key] = facts
    _fact_cache_ttl[key] = time.time()



def retrieve_from_graph(query_analysis, top_k=20, max_depth=2):
    keywords = query_analysis.get("keywords", [])
    entities = query_analysis.get("entities", [])
    facts = []
    seen_fact_ids = set()

    for kw in keywords:
        related = get_related_keywords(kw, min_weight=0.3)
        for rel_kw, weight in related:
            facts.extend(get_facts_by_keyword(rel_kw))
        facts.extend(get_facts_by_keyword(kw))

    # Entity-based retrieval: handle both strings and dicts
    for ent in entities:
        if isinstance(ent, dict):
            ent_name = ent.get("text") or ent.get("name") or ent.get("entity") or ""
        else:
            ent_name = str(ent)
        if not ent_name:
            continue
        conn = db.db_connect("external_graph")
        cur = conn.cursor()
        cur.execute("SELECT global_node_id, canonical_name FROM global_nodes WHERE canonical_name=? OR EXISTS (SELECT 1 FROM json_each(global_nodes.aliases_json) WHERE value = ?) LIMIT 1",
                    (ent_name, ent_name))
        row = cur.fetchone()
        if row:
            gid, canonical = row
            edges = get_global_node_edges(gid)
            for edge in edges:
                other_gid = edge["source_node_id"] if edge["source_node_id"] != gid else edge["target_node_id"]
                cur.execute("SELECT canonical_name, node_type FROM global_nodes WHERE global_node_id=?", (other_gid,))
                other = cur.fetchone()
                if other:
                    facts.extend(get_facts_by_keyword(other[0]))
        conn.close()

    unique_facts = []
    for fact in facts:
        fid = fact.get("fact_id")
        if fid and fid not in seen_fact_ids:
            seen_fact_ids.add(fid)
            unique_facts.append(fact)

    unique_facts.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    # Ensure 'result' is defined (should be, but safety)
    if 'result' not in locals():
        result = []

    # Optional Recoll full-text search (additional source)
    recoll_facts = []
    if config.USE_RECOLL:
        try:
            from core.recoll_client import RecollClient
            recoll_client = RecollClient()
            recoll_results, _ = recoll_client.search(query_analysis.get('original', ''), limit=5, fetch_text=False)
            for doc in recoll_results:
                file_url = doc.get("path") or doc.get("url", "")
                file_path = file_url.replace("file://", "")
                if file_path:
                    conn = db.db_connect("index")
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT dc.chunk_id, dc.doc_hash, dc.chunk_text
                        FROM document_chunks dc
                        JOIN documents d ON dc.doc_hash = d.file_hash
                        WHERE d.file_path = ?
                        LIMIT 3
                    """, (file_path,))
                    rows = cur.fetchall()
                    conn.close()
                    for row in rows:
                        recoll_facts.append({
                            "fact_id": None,
                            "doc_hash": row["doc_hash"],
                            "doc_name": file_path,
                            "fact_text": row["chunk_text"][:300],
                            "canonical_value": "",
                            "source_span": "",
                            "confidence": 0.6,
                            "chunk_id": row["chunk_id"],
                        })
            recoll_client.close()
        except ImportError:
            pass
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Recoll fallback error: {e})")

    return (unique_facts + recoll_facts)[:top_k]


def fallback_to_chunks(query, top_k=None):
    if top_k is None:
        top_k = config.CHAT_TOP_K_CHUNKS
    q_emb = get_embedding(query)
    if not q_emb:
        return []
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, doc_hash, chunk_text, embedding FROM chunk_embeddings LIMIT ?", (top_k * 10,))
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return []
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    chunk_data = [(r[0], r[1], r[2], r[3]) for r in rows]
    embeddings = [np.frombuffer(r[3], dtype=np.float32) for r in rows]
    matrix = np.vstack(embeddings)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-8
    sims = matrix @ q / (norms * (q_norm + 1e-8))
    results = [
        (float(sim), chunk_id, doc_hash, chunk_text)
        for sim, (chunk_id, doc_hash, chunk_text, _) in zip(sims, chunk_data)
    ]
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def re_rank_facts(facts, query, top_k=None):
    """Re-rank facts using memory and logic modules for better relevance."""
    if not facts:
        return facts
    # Get relevant logic modules
    logic_mods = retrieve_logic_modules(query, top_k=3)
    logic_keywords = set()
    for sim, lid, name, category, summary, content in logic_mods:
        for word in summary.split()[:5]:
            if len(word) > 3:
                logic_keywords.add(word.lower())
    # Get recent memories
    memories = retrieve_memories(query, top_k=3)
    memory_keywords = set()
    for sim, mid, content, mtype in memories:
        for word in content.split()[:10]:
            if len(word) > 3:
                memory_keywords.add(word.lower())
    # Score facts based on keyword overlap with logic/memory
    for fact in facts:
        fact_text = fact.get("fact_text", "").lower()
        score = fact.get("confidence", 0)
        overlap = len(set(fact_text.split()) & logic_keywords) + len(set(fact_text.split()) & memory_keywords)
        fact["_relevance_boost"] = overlap * 0.05
        fact["_final_score"] = score + fact["_relevance_boost"]
    facts.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
    if top_k:
        return facts[:top_k]
    return facts