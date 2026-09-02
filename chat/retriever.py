
import numpy as np
import time
from core import db
from core.embeddings import get_embedding
from core.text_utils import tokenize
from graph.graph_queries import get_related_keywords, get_facts_by_keyword, get_global_node_edges
from graph.expansion import expand_facts_via_multi_hop
from logic.retrieve import retrieve_logic_modules
from memory.retrieve import retrieve_memories
import config
from fuzzywuzzy import fuzz

_vector_store_cache = {}
_vector_store_cache_ttl = 300

def _get_vector_store(model):
    now = time.time()
    if model in _vector_store_cache:
        entry = _vector_store_cache[model]
        if now - entry["ts"] < _vector_store_cache_ttl:
            return entry["store"]
    from core.vector_store import ExactVectorStore
    store = ExactVectorStore(config.EMBEDDINGS_DB_FILE, "chunk_embeddings", "chunk_id", "embedding")
    _vector_store_cache[model] = {"store": store, "ts": now}
    return store


def retrieve_from_graph(query_analysis, top_k=None, max_depth=2, debug=False):
    keywords = query_analysis.get("keywords", [])
    entities = query_analysis.get("entities", [])
    facts = []
    seen_fact_ids = set()

    for kw in keywords:
        related = get_related_keywords(kw, min_weight=0.3)
        for rel_kw, weight in related:
            facts.extend(get_facts_by_keyword(rel_kw))
        facts.extend(get_facts_by_keyword(kw))

    for ent in entities:
        ent_name = ent.get("text") if isinstance(ent, dict) else str(ent)
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

    if top_k is None:
        return unique_facts
    return unique_facts[:top_k]


def fallback_to_chunks(query, top_k=None, debug=False):
    if top_k is None:
        top_k = config.CHAT_TOP_K_CHUNKS

    results = []
    seen = set()

    try:
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
            cur.execute(f"SELECT chunk_id, doc_hash, chunk_text FROM document_chunks WHERE {like_sql} LIMIT 500", params)
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

    model = config.EMBEDDING_ENDPOINTS[0]["model"]
    q_emb = get_embedding(query, model=model, space='hyperbolic')
    if q_emb is not None:
        try:
            store = _get_vector_store(model)
            if store.tree is not None:
                top_distances = store.tree.search(q_emb, k=config.CHAT_TOP_K_CHUNKS * 5)
                ids = [id_ for id_, _ in top_distances]
                if ids:
                    conn = db.db_connect("index")
                    cur = conn.cursor()
                    placeholders = ",".join("?" for _ in ids)
                    cur.execute(f"SELECT chunk_id, doc_hash, chunk_text FROM document_chunks WHERE chunk_id IN ({placeholders})", ids)
                    rows = cur.fetchall()
                    conn.close()
                    id_to_row = {row["chunk_id"]: row for row in rows}
                    for cid, dist in top_distances:
                        if cid in id_to_row and cid not in seen:
                            seen.add(cid)
                            row = id_to_row[cid]
                            sim = 1.0 / (1.0 + dist)
                            results.append((sim * 0.7, cid, row["doc_hash"], row["chunk_text"]))
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Embedding chunk fallback error: {e})")

    final = []
    seen_ids = set()
    for score, chunk_id, doc_hash, text in results:
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        final.append((score, chunk_id, doc_hash, text))
    final.sort(key=lambda x: x[0], reverse=True)
    if debug:
        print(f"  Chunk retrieval: {len(final)} results")
    return final[:top_k]


def re_rank_facts(facts, query, top_k=None):
    if not facts:
        return facts
    try:
        query_tokens = set(tokenize(query))
    except Exception:
        query_tokens = set(query.lower().split())

    logic_mods = retrieve_logic_modules(query, top_k=3)
    logic_keywords = set()
    for sim, lid, name, category, summary, content in logic_mods:
        for word in summary.split()[:5]:
            if len(word) > 3:
                logic_keywords.add(word.lower())

    memories = retrieve_memories(query, top_k=3)
    memory_keywords = set()
    for sim, mid, content, mtype in memories:
        for word in content.split()[:10]:
            if len(word) > 3:
                memory_keywords.add(word.lower())

    for fact in facts:
        fact_text = fact.get("fact_text", "").lower()
        canonical = fact.get("canonical_value", "").lower()
        fact_tokens = set(tokenize(fact_text)) | set(tokenize(canonical))
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
