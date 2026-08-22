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

def store_kg_triple(subject, predicate, object_, source_document_id=None, confidence=0.0):
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO kg_triples (subject, predicate, object, source_document_id, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (subject, predicate, object_, source_document_id, confidence))
    triple_id = cur.lastrowid
    conn.commit(); conn.close()
    return triple_id

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
