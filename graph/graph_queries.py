from core import db

def get_related_keywords(keyword, min_weight=0.5):
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT kw_b, weight FROM keyword_cooccurrence WHERE kw_a=? AND weight>=?
        UNION ALL
        SELECT kw_a, weight FROM keyword_cooccurrence WHERE kw_b=? AND weight>=?
    """, (keyword, min_weight, keyword, min_weight))
    rows = cur.fetchall()
    conn.close()
    return [(row[0], row[1]) for row in rows]


def get_facts_by_keyword(keyword):
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("""
        SELECT f.fact_id, f.doc_hash, f.doc_name, f.fact_type, f.fact_text,
               f.canonical_value, f.source_span, f.confidence, fs.chunk_id
        FROM key_facts f
        LEFT JOIN fact_sources fs ON f.fact_id = fs.fact_id
        WHERE f.fact_text LIKE ? OR f.canonical_value LIKE ?
        ORDER BY f.confidence DESC
    """, (f"%{keyword}%", f"%{keyword}%"))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_global_node_edges(global_node_id):
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT e.edge_id, e.source_node_id, e.target_node_id, e.relation_type, e.weight, e.doc_hash, e.source_span, e.confidence
        FROM global_edges e
        WHERE e.source_node_id=? OR e.target_node_id=?
        ORDER BY e.weight DESC
    """, (global_node_id, global_node_id))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]
