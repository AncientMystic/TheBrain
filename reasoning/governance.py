from core import db


def quality_gate(claim):
    if not claim.get("source_span") or len(claim.get("source_span", "").split()) > 4:
        return False
    if not claim.get("subject") or not claim.get("predicate"):
        return False
    if claim.get("confidence", 0) < 0.5:
        return False
    return True


def detect_contradictions():
    """
    Detect contradictions across kg_triples, key_facts, and external_graph.
    Returns list of dicts describing conflicts.
    """
    contradictions = []

    # 1. kg_triples
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id as id_a, b.id as id_b, a.subject, a.predicate, a.object as obj_a, b.object as obj_b,
               a.confidence as conf_a, b.confidence as conf_b
        FROM kg_triples a
        JOIN kg_triples b ON a.subject = b.subject AND a.predicate = b.predicate AND a.id < b.id
        WHERE a.object != b.object
    """)
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        contradictions.append({
            "source": "kg_triples",
            "triple_a_id": row["id_a"],
            "triple_b_id": row["id_b"],
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object_a": row["obj_a"],
            "object_b": row["obj_b"],
            "confidence_a": row["conf_a"],
            "confidence_b": row["conf_b"],
        })

    # 2. key_facts
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("""
        SELECT f1.fact_id as id1, f2.fact_id as id2, f1.canonical_value as sub1, f2.canonical_value as sub2,
               f1.fact_text as text1, f2.fact_text as text2,
               f1.confidence as conf1, f2.confidence as conf2
        FROM key_facts f1
        JOIN key_facts f2 ON f1.fact_type = f2.fact_type AND f1.fact_id < f2.fact_id
        WHERE f1.canonical_value IS NOT NULL AND f2.canonical_value IS NOT NULL
          AND f1.canonical_value = f2.canonical_value
          AND f1.fact_text != f2.fact_text
    """)
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        contradictions.append({
            "source": "key_facts",
            "fact_a_id": row["id1"],
            "fact_b_id": row["id2"],
            "canonical_value": row["sub1"],
            "text_a": row["text1"],
            "text_b": row["text2"],
            "confidence_a": row["conf1"],
            "confidence_b": row["conf2"],
        })

    # 3. external_graph
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT e1.edge_id as id1, e2.edge_id as id2, e1.source_node_id, e1.relation_type,
               n1.canonical_name as source_name, n2a.canonical_name as target_a, n2b.canonical_name as target_b,
               e1.confidence as conf1, e2.confidence as conf2
        FROM global_edges e1
        JOIN global_edges e2 ON e1.source_node_id = e2.source_node_id
                            AND e1.relation_type = e2.relation_type
                            AND e1.edge_id < e2.edge_id
        JOIN global_nodes n1 ON e1.source_node_id = n1.global_node_id
        JOIN global_nodes n2a ON e1.target_node_id = n2a.global_node_id
        JOIN global_nodes n2b ON e2.target_node_id = n2b.global_node_id
        WHERE e1.target_node_id != e2.target_node_id
    """)
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        contradictions.append({
            "source": "external_graph",
            "edge_a_id": row["id1"],
            "edge_b_id": row["id2"],
            "source_name": row["source_name"],
            "relation_type": row["relation_type"],
            "target_a": row["target_a"],
            "target_b": row["target_b"],
            "confidence_a": row["conf1"],
            "confidence_b": row["conf2"],
        })

    return contradictions


def resolve_contradictions(contradictions):
    """
    Auto-resolve contradictions by deleting the lower-confidence triple or fact.
    For external_graph, we delete the edge with lower confidence.
    Returns list of resolutions.
    """
    resolutions = []
    for c in contradictions:
        source = c.get("source")
        if source == "kg_triples":
            id_a = c["triple_a_id"]
            id_b = c["triple_b_id"]
            conf_a = c.get("confidence_a", 0.0) or 0.0
            conf_b = c.get("confidence_b", 0.0) or 0.0
            delete_id = id_a if conf_a < conf_b else id_b
            conn = db.db_connect("reasoning")
            conn.execute("DELETE FROM kg_triples WHERE id=?", (delete_id,))
            conn.commit()
            conn.close()
            resolutions.append(f"Resolved kg_triples contradiction: deleted id {delete_id} (lower confidence)")

        elif source == "key_facts":
            id1 = c["fact_a_id"]
            id2 = c["fact_b_id"]
            conf1 = c.get("confidence_a", 0.0) or 0.0
            conf2 = c.get("confidence_b", 0.0) or 0.0
            delete_id = id1 if conf1 < conf2 else id2
            conn = db.db_connect("key_facts")
            conn.execute("DELETE FROM key_facts WHERE fact_id=?", (delete_id,))
            conn.commit()
            conn.close()
            resolutions.append(f"Resolved key_facts contradiction: deleted fact_id {delete_id}")

        elif source == "external_graph":
            id1 = c["edge_a_id"]
            id2 = c["edge_b_id"]
            conf1 = c.get("confidence_a", 0.0) or 0.0
            conf2 = c.get("confidence_b", 0.0) or 0.0
            delete_id = id1 if conf1 < conf2 else id2
            conn = db.db_connect("external_graph")
            conn.execute("DELETE FROM global_edges WHERE edge_id=?", (delete_id,))
            conn.commit()
            conn.close()
            resolutions.append(f"Resolved external_graph contradiction: deleted edge_id {delete_id}")

    return resolutions


def compute_confidence(verification_results):
    weights = {'text_grounding': 0.1, 'symstep': 0.4, 'vericot': 0.3, 'fidelis': 0.2, 'rcot': 0.1}
    score = 0.0
    for result in verification_results:
        if result.get('verified'):
            layer = result.get('layer', '')
            score += weights.get(layer, 0.0) * result.get('confidence', 0.0)
    return min(score, 1.0)


def store_provenance(step_id, grounding_type, kg_triple_id=None, text_span_id=None, prior_step_id=None, confidence=0.0):
    conn = db.db_connect("reasoning")
    conn.execute("""
        INSERT INTO grounding_records (reasoning_node_id, grounding_type, kg_triple_id, text_span_id, prior_step_id, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (step_id, grounding_type, kg_triple_id, text_span_id, prior_step_id, confidence))
    conn.commit()
    conn.close()


def audit_reasoning_path(query_id):
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("SELECT * FROM reasoning_nodes WHERE query_id=? ORDER BY step_number", (query_id,))
    nodes = [dict(row) for row in cur.fetchall()]
    conn.close()
    return nodes
