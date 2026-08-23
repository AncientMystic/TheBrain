import json
from core import db

def create_reasoning_node(query_id, step_number, node_type, content, formal_repr=None, confidence=0.0):
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reasoning_nodes (query_id, step_number, node_type, content, formal_representation, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (query_id, step_number, node_type, content, formal_repr, confidence))
    node_id = cur.lastrowid
    conn.commit(); conn.close()
    return node_id

def add_reasoning_edge(source_id, target_id, relation_type, verified=0):
    conn = db.db_connect("reasoning")
    conn.execute("""
        INSERT INTO reasoning_edges (source_node_id, target_node_id, relation_type, verified)
        VALUES (?, ?, ?, ?)
    """, (source_id, target_id, relation_type, verified))
    conn.commit(); conn.close()

def get_reasoning_path(query_id):
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("SELECT * FROM reasoning_nodes WHERE query_id=? ORDER BY step_number", (query_id,))
    nodes = [dict(row) for row in cur.fetchall()]
    conn.close()
    return nodes

def rebuild_implied_triples():
    """Rebuild materialized transitive closure from kg_triples."""
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("SELECT subject, predicate, object FROM kg_triples")
    triples = [tuple(row) for row in cur.fetchall()]
    # Simple transitive closure for transitive predicates
    transitive = {"is_a", "part_of", "located_in", "belongs_to", "works_for"}
    closure = set(triples)
    changed = True
    while changed:
        changed = False
        for s1, p1, o1 in list(closure):
            if p1 not in transitive:
                continue
            for s2, p2, o2 in list(closure):
                if s2 == o1 and p2 == p1:
                    new_triple = (s1, p1, o2)
                    if new_triple not in closure:
                        closure.add(new_triple)
                        changed = True
    # Clear and repopulate implied_triples
    cur.execute("DELETE FROM implied_triples")
    for s, p, o in closure:
        cur.execute("INSERT OR IGNORE INTO implied_triples (subject, predicate, object) VALUES (?, ?, ?)",
                    (s, p, o))
    conn.commit(); conn.close()

def store_kg_triples_batch(triples):
    """Insert multiple kg_triples and rebuild closure once."""
    if not triples:
        return
    conn = db.db_connect("reasoning")
    conn.executemany("""
        INSERT INTO kg_triples (subject, predicate, object, source_document_id, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, triples)
    conn.commit(); conn.close()
    rebuild_implied_triples()


def store_kg_triple(subject, predicate, object_, source_document_id=None, confidence=0.0):
    return store_kg_triples_batch([(subject, predicate, object_, source_document_id, confidence)])

def query_kg_triples(subject=None, predicate=None, object_=None):
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    sql = "SELECT * FROM kg_triples WHERE 1=1"
    params = []
    if subject:
        sql += " AND subject=?"
        params.append(subject)
    if predicate:
        sql += " AND predicate=?"
        params.append(predicate)
    if object_:
        sql += " AND object=?"
        params.append(object_)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def link_grounding(step_id, grounding_type, kg_triple_id=None, text_span_id=None, prior_step_id=None, confidence=0.0):
    conn = db.db_connect("reasoning")
    conn.execute("""
        INSERT INTO grounding_records (reasoning_node_id, grounding_type, kg_triple_id, text_span_id, prior_step_id, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (step_id, grounding_type, kg_triple_id, text_span_id, prior_step_id, confidence))
    conn.commit(); conn.close()
