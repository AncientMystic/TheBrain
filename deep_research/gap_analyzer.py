"""
Analyze internal databases to find knowledge gaps.
Used by autonomous Recoll-guided learning.
"""
import config
from core import db

def find_low_confidence_facts(threshold=0.6, limit=20):
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("""
        SELECT fact_id, doc_hash, doc_name, fact_text, canonical_value, confidence
        FROM key_facts
        WHERE confidence < ?
        ORDER BY confidence ASC
        LIMIT ?
    """, (threshold, limit))
    rows = cur.fetchall()
    conn.close()
    gaps = []
    for row in rows:
        gaps.append({
            "type": "low_confidence",
            "entity": row["canonical_value"] or row["fact_text"][:50],
            "fact_id": row["fact_id"],
            "text": row["fact_text"],
            "confidence": row["confidence"],
        })
    return gaps

def find_entities_with_few_edges(min_edges=2, limit=20):
    """Find global nodes with few relationships."""
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT gn.global_node_id, gn.canonical_name,
               (SELECT COUNT(*) FROM global_edges ge WHERE ge.source_node_id = gn.global_node_id OR ge.target_node_id = gn.global_node_id) AS edge_count
        FROM global_nodes gn
        WHERE edge_count < ?
        ORDER BY edge_count ASC
        LIMIT ?
    """, (min_edges, limit))
    rows = cur.fetchall()
    conn.close()
    gaps = []
    for row in rows:
        gaps.append({
            "type": "sparse_entity",
            "entity": row["canonical_name"],
            "global_node_id": row["global_node_id"],
            "edge_count": row["edge_count"],
        })
    return gaps

def find_keywords_with_low_coverage(min_facts=3, limit=20):
    """Find keywords with few associated facts (using separate DB connections)."""
    # 1. Get keywords from external_graph.db
    conn_eg = db.db_connect("external_graph")
    cur_eg = conn_eg.cursor()
    cur_eg.execute("SELECT keyword FROM keyword_topic_edges GROUP BY keyword")
    keyword_rows = cur_eg.fetchall()
    conn_eg.close()

    # 2. Count facts for each keyword in key_facts.db
    gaps = []
    conn_kf = db.db_connect("key_facts")
    cur_kf = conn_kf.cursor()

    for row in keyword_rows:
        keyword = row[0]
        try:
            cur_kf.execute("""
                SELECT COUNT(*) FROM key_facts
                WHERE canonical_value = ? OR fact_text LIKE ?
            """, (keyword, f"%{keyword}%"))
            count = cur_kf.fetchone()[0]
        except Exception:
            count = 0
        if count < min_facts:
            gaps.append({
                "type": "low_coverage_keyword",
                "entity": keyword,
                "fact_count": count,
            })
        if len(gaps) >= limit:
            break

    conn_kf.close()
    return gaps

def find_unconnected_topics(limit=20):
    """Find topics with no cross-document links."""
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT gn.global_node_id, gn.canonical_name
        FROM global_nodes gn
        LEFT JOIN cross_doc_links cdl ON gn.global_node_id = cdl.global_node_id
        WHERE cdl.link_id IS NULL
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    gaps = []
    for row in rows:
        gaps.append({
            "type": "unconnected_topic",
            "entity": row["canonical_name"],
            "global_node_id": row["global_node_id"],
        })
    return gaps

def find_missing_quotes(limit=20):
    """Find documents that have facts but no associated quotes."""
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT d.doc_hash, d.filename
            FROM documents d
            LEFT JOIN quotes q ON d.file_hash = q.doc_hash
            WHERE q.quote_id IS NULL
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
    except Exception:
        rows = []
    conn.close()
    return [{"type": "missing_quotes", "entity": r["filename"], "global_node_id": r["doc_hash"]} for r in rows]


def find_missing_relationships(limit=20):
    """Find global nodes with no edges."""
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT gn.global_node_id, gn.canonical_name
        FROM global_nodes gn
        LEFT JOIN global_edges ge ON gn.global_node_id = ge.source_node_id OR gn.global_node_id = ge.target_node_id
        WHERE ge.edge_id IS NULL
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [{"type": "missing_relationships", "entity": r["canonical_name"], "global_node_id": r["global_node_id"]} for r in rows]


def find_under_explored_topics(min_edges=3, limit=20):
    """Find topics with fewer than min_edges edges."""
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT gn.global_node_id, gn.canonical_name,
               (SELECT COUNT(*) FROM global_edges ge WHERE ge.source_node_id = gn.global_node_id OR ge.target_node_id = gn.global_node_id) AS edge_count
        FROM global_nodes gn
        WHERE edge_count < ?
        ORDER BY edge_count ASC
        LIMIT ?
    """, (min_edges, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"type": "under_explored_topic", "entity": r["canonical_name"], "global_node_id": r["global_node_id"], "edge_count": r["edge_count"]} for r in rows]


def get_all_gaps(limit_per_type=20):
    """Return a combined list of knowledge gaps."""
    gaps = []
    gaps.extend(find_low_confidence_facts(limit=limit_per_type))
    gaps.extend(find_entities_with_few_edges(limit=limit_per_type))
    gaps.extend(find_keywords_with_low_coverage(limit=limit_per_type))
    gaps.extend(find_unconnected_topics(limit=limit_per_type))
    gaps.extend(find_missing_relationships(limit=limit_per_type))
    gaps.extend(find_under_explored_topics(limit=limit_per_type))
    gaps.extend(find_missing_quotes(limit=limit_per_type))
    return gaps
