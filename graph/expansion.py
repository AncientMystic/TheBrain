"""
Multi-hop graph expansion and embedding-based retrieval enhancements.
"""
import numpy as np
from core import db
from core.embeddings import get_embedding
from graph.graph_queries import get_related_keywords, get_facts_by_keyword, get_global_node_edges
import config


def expand_facts_via_multi_hop(initial_facts, max_depth=2, max_facts=200):
    """
    Expand a list of facts by exploring related entities in the external graph.
    Uses both keyword co-occurrence and global node edges.
    """
    all_facts = list(initial_facts)
    seen_ids = {f.get("fact_id") for f in all_facts if f.get("fact_id")}
    current_frontier = initial_facts[:50]  # limit initial expansion
    analysis_cache = {}

    for depth in range(max_depth):
        new_facts = []
        for fact in current_frontier:
            # get keywords from fact text and canonical value using stopword-filtered tokens
            from core.text_utils import tokenize, get_bigrams
            text = fact.get("fact_text", "")
            val = fact.get("canonical_value", "")
            combined = text + " " + val
            tokens = tokenize(combined)
            keywords = tokens[:5] + list(get_bigrams(tokens))[:3]
            conn = db.db_connect("external_graph")
            cur = conn.cursor()
            for kw in keywords:
                # related keywords via co-occurrence
                for rel_kw, _ in get_related_keywords(kw, min_weight=0.3):
                    for f in get_facts_by_keyword(rel_kw, limit=20):
                        if f.get("fact_id") not in seen_ids:
                            new_facts.append(f)
                            seen_ids.add(f.get("fact_id"))
                # direct keyword facts
                for f in get_facts_by_keyword(kw, limit=20):
                    if f.get("fact_id") not in seen_ids:
                        new_facts.append(f)
                        seen_ids.add(f.get("fact_id"))

            # expand via global graph nodes using same connection
            from chat.query_analyzer import analyze_query
            if text not in analysis_cache:
                analysis_cache[text] = analyze_query(text)
            analysis = analysis_cache[text]
            for ent in analysis.get("entities", []):
                ent_name = ent.get("text") if isinstance(ent, dict) else str(ent)
                if not ent_name:
                    continue
                # Use cached connection already open; just execute
                cur.execute("""
                    SELECT global_node_id FROM global_nodes
                    WHERE canonical_name=? OR EXISTS (SELECT 1 FROM json_each(global_nodes.aliases_json) WHERE value=?)
                    LIMIT 1
                """, (ent_name, ent_name))
                row = cur.fetchone()
                if row:
                    gid = row[0]
                    edges = get_global_node_edges(gid)
                    for edge in edges:
                        other_gid = edge["source_node_id"] if edge["source_node_id"] != gid else edge["target_node_id"]
                        cur.execute("SELECT canonical_name FROM global_nodes WHERE global_node_id=?", (other_gid,))
                        other = cur.fetchone()
                        if other:
                            for f in get_facts_by_keyword(other[0], limit=20):
                                if f.get("fact_id") not in seen_ids:
                                    new_facts.append(f)
                                    seen_ids.add(f.get("fact_id"))
            conn.close()
        if not new_facts:
            break
        all_facts.extend(new_facts)
        current_frontier = new_facts[:50]  # limit next frontier
        if len(all_facts) >= max_facts:
            break
    return all_facts[:max_facts]


def embed_based_retrieval(query, top_k=20):
    """
    Retrieve facts/chunks based on embedding similarity to the query.
    Falls back to chunk search.
    """
    from chat.retriever import fallback_to_chunks
    chunks = fallback_to_chunks(query, top_k=top_k)
    return chunks


def get_fact_ids_by_entity(entity_name):
    """Retrieve fact IDs associated with a normalized entity name."""
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("SELECT fact_id FROM entity_fact_index WHERE normalized_name=?", (entity_name,))
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]

def expand_facts_via_entity_index(facts):
    """Use precomputed entity index to find related facts."""
    new_facts = []
    seen = {f.get("fact_id") for f in facts if f.get("fact_id")}
    for fact in facts:
        # Get entities for this fact from entity_fact_index (simple approach: use canonical_value)
        # But we need entity names; we can query by fact_id
        conn = db.db_connect("key_facts")
        cur = conn.cursor()
        cur.execute("SELECT entity_name FROM entity_fact_index WHERE fact_id=?", (fact.get("fact_id"),))
        entity_rows = cur.fetchall()
        conn.close()
        for row in entity_rows:
            entity = row[0]
            fact_ids = get_fact_ids_by_entity(entity)
            for fid in fact_ids:
                if fid not in seen:
                    # Load fact
                    conn = db.db_connect("key_facts")
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM key_facts WHERE fact_id=?", (fid,))
                    fact_row = cur.fetchone()
                    conn.close()
                    if fact_row:
                        new_facts.append(dict(fact_row))
                        seen.add(fid)
    return new_facts
