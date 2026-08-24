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

_chunk_matrix_cache = {}
_chunk_matrix_cache_ttl = 300  # seconds

def _get_chunk_matrix(model):
    """Return cached (matrix, rows) for a given embedding model."""
    now = time.time()
    if model in _chunk_matrix_cache:
        entry = _chunk_matrix_cache[model]
        if now - entry["ts"] < _chunk_matrix_cache_ttl:
            return entry["matrix"], entry["rows"]

    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, doc_hash, chunk_text, embedding FROM chunk_embeddings WHERE model=?",
        (model,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None, []

    matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    _chunk_matrix_cache[model] = {"matrix": matrix, "rows": rows, "ts": now}
    return matrix, rows

from fuzzywuzzy import fuzz

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

    unique_facts = re_rank_facts(unique_facts, query_analysis.get('original', ''))

    # Keep only query-relevant facts, with a small confidence fallback
    try:
        from core.text_utils import tokenize
        query_tokens = set(tokenize(query_analysis.get('original', '')))
    except Exception:
        query_tokens = set((query_analysis.get('original', '') or '').lower().split())

    relevant_facts = []
    for f in unique_facts:
        fact_text = (f.get('fact_text') or '').lower()
        canonical = (f.get('canonical_value') or '').lower()
        try:
            fact_tokens = set(tokenize(fact_text)) | set(tokenize(canonical))
        except Exception:
            fact_tokens = set(fact_text.split()) | set(canonical.split())

        has_query_overlap = bool(query_tokens & fact_tokens)
        has_high_fuzzy = f.get('_fuzzy_score', 0) >= 70

        if has_query_overlap or has_high_fuzzy:
            relevant_facts.append(f)

    if relevant_facts:
        unique_facts = relevant_facts[:top_k]
    else:
        unique_facts = unique_facts[:10]

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
    """Hybrid chunk retrieval: ranked keyword matches + embedding similarity."""
    if top_k is None:
        top_k = config.CHAT_TOP_K_CHUNKS

    results = []
    seen = set()

    # Token variants for keyword search
    try:
        from core.text_utils import tokenize
        tokens = tokenize(query)
        variants = []
        for t in tokens:
            variants.append(t)
            if t.endswith("s") and len(t) > 3:
                variants.append(t[:-1])
            if t.endswith("ies") and len(t) > 4:
                variants.append(t[:-3] + "y")
    except Exception:
        tokens = [w.lower() for w in query.split() if len(w) > 2]
        variants = tokens

    if variants:
        try:
            conn = db.db_connect("index")
            cur = conn.cursor()
            likes = []
            params = []
            for v in variants:
                likes.append("chunk_text LIKE ?")
                params.append(f"%{v}%")
            like_sql = " OR ".join(likes)
            cur.execute(
                f"SELECT chunk_id, doc_hash, chunk_text FROM document_chunks WHERE {like_sql} LIMIT 500",
                params,
            )
            rows = cur.fetchall()
            conn.close()

            phrase = " ".join(variants)
            for row in rows:
                chunk_id = row["chunk_id"]
                text = row["chunk_text"].lower()
                score = 0.0
                matched = 0
                for v in variants:
                    if v in text:
                        score += max(1.0, len(v) / 4.0)
                        matched += 1
                if matched == 0:
                    continue
                if phrase in text:
                    score += len(variants) * 2.0
                results.append((score, chunk_id, row["doc_hash"], row["chunk_text"]))
                seen.add(chunk_id)
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Keyword chunk fallback error: {e})")

    # Embedding similarity hybrid, using precomputed matrix when possible
    model = config.EMBEDDING_ENDPOINTS[0]["model"]
    q_emb = get_embedding(query, model=model)
    if q_emb:
        try:
            matrix, rows = _get_chunk_matrix(model)
            if matrix is not None and rows:
                q = np.array(q_emb, dtype=np.float32)
                q_norm = np.linalg.norm(q)
                norms = np.linalg.norm(matrix, axis=1)
                norms[norms == 0] = 1e-8
                sims = matrix @ q / (norms * (q_norm + 1e-8))

                for row, sim in zip(rows, sims):
                    chunk_id = row["chunk_id"]
                    doc_hash = row["doc_hash"]
                    chunk_text = row["chunk_text"]
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        results.append((float(sim) * 0.5, chunk_id, doc_hash, chunk_text))
                    else:
                        for idx, item in enumerate(results):
                            if item[1] == chunk_id:
                                old_score, _, _, _ = item
                                combined = old_score + float(sim) * 0.5
                                results[idx] = (combined, chunk_id, doc_hash, chunk_text)
                                break
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Embedding fallback error: {e})")

    # Deduplicate and sort by combined score
    final = []
    seen_ids = set()
    for score, chunk_id, doc_hash, text in results:
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        final.append((score, chunk_id, doc_hash, text))
    final.sort(key=lambda x: x[0], reverse=True)
    return final[:top_k]
def re_rank_facts(facts, query, top_k=None):
    """Re-rank facts using query token overlap, memory, and logic modules."""
    if not facts:
        return facts

    try:
        from core.text_utils import tokenize
        query_tokens = set(tokenize(query))
    except Exception:
        query_tokens = set(query.lower().split())

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

    # Score facts based on query token overlap, logic, and memory
    for fact in facts:
        fact_text = fact.get("fact_text", "").lower()
        canonical = fact.get("canonical_value", "").lower()

        try:
            from core.text_utils import tokenize
            fact_tokens = set(tokenize(fact_text)) | set(tokenize(canonical))
        except Exception:
            fact_tokens = set(fact_text.split()) | set(canonical.split())

        score = fact.get("confidence", 0)
        query_overlap = len(fact_tokens & query_tokens)
        logic_overlap = len(fact_tokens & logic_keywords)
        memory_overlap = len(fact_tokens & memory_keywords)

        fact["_relevance_boost"] = query_overlap * 0.25 + logic_overlap * 0.05 + memory_overlap * 0.05
        fact["_final_score"] = score + fact["_relevance_boost"]

    facts.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
    if top_k:
        return facts[:top_k]
    return facts
