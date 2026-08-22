from core import db
from core.embeddings import get_embedding
from graph.graph_queries import get_related_keywords, get_facts_by_keyword, get_global_node_edges
import numpy as np
import config


def retrieve_from_graph(query_analysis, top_k=20):
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
        cur.execute("SELECT global_node_id, canonical_name FROM global_nodes WHERE canonical_name=? OR aliases_json LIKE ? LIMIT 1",
                    (ent_name, f'%"{ent_name}"%'))
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
    return unique_facts[:top_k]


def fallback_to_chunks(query, top_k=None):
    if top_k is None:
        top_k = config.CHAT_TOP_K_CHUNKS
    q_emb = get_embedding(query)
    if not q_emb:
        return []
    conn = db.db_connect("embeddings")
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, doc_hash, chunk_text, embedding FROM chunk_embeddings")
    rows = cur.fetchall()
    conn.close()
    results = []
    q = np.array(q_emb, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    for chunk_id, doc_hash, chunk_text, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32)
        sim = float(np.dot(q, emb) / (q_norm * np.linalg.norm(emb) + 1e-8))
        results.append((sim, chunk_id, doc_hash, chunk_text))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]
