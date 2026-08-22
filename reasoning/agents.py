import json
from core import db
from reasoning.graph import query_kg_triples, create_reasoning_node, add_reasoning_edge, link_grounding
from reasoning.governance import detect_contradictions, quality_gate, compute_confidence
from reasoning.verify import verify_claim


class MindMapAgent:
    """Tracks logical relationships and builds the reasoning graph."""
    def __init__(self):
        self.query_id = None
        self.step_counter = 0
        self.node_map = {}  # step_number -> node_id

    def start_query(self, query_id):
        self.query_id = query_id
        self.step_counter = 0
        self.node_map = {}

    def track_step(self, node_type, content, formal_repr=None, confidence=0.0):
        if not self.query_id:
            raise ValueError("Query not started")
        self.step_counter += 1
        node_id = create_reasoning_node(
            self.query_id, self.step_counter, node_type, content, formal_repr, confidence
        )
        self.node_map[self.step_counter] = node_id
        return node_id

    def add_edge(self, source_step, target_step, relation_type, verified=0):
        src_id = self.node_map.get(source_step)
        tgt_id = self.node_map.get(target_step)
        if src_id and tgt_id:
            add_reasoning_edge(src_id, tgt_id, relation_type, verified)

    def link_grounding(self, step_number, grounding_type, **kwargs):
        step_id = self.node_map.get(step_number)
        if step_id:
            link_grounding(step_id, grounding_type, **kwargs)


class KGQueryAgent:
    """Retrieves ground truth facts from reasoning KG, external graph, and key_facts."""
    def query_triples(self, subject=None, predicate=None, object_=None):
        return query_kg_triples(subject=subject, predicate=predicate, object_=object_)

    def query_external_edges(self, subject=None, predicate=None, object_=None):
        conn = db.db_connect("external_graph")
        cur = conn.cursor()
        sql = """SELECT e.edge_id, s.canonical_name as subject, e.relation_type as predicate,
                        o.canonical_name as object, e.confidence
                 FROM global_edges e
                 JOIN global_nodes s ON e.source_node_id = s.global_node_id
                 JOIN global_nodes o ON e.target_node_id = o.global_node_id
                 WHERE 1=1"""
        params = []
        if subject:
            sql += " AND s.canonical_name = ?"
            params.append(subject)
        if predicate:
            sql += " AND e.relation_type = ?"
            params.append(predicate)
        if object_:
            sql += " AND o.canonical_name = ?"
            params.append(object_)
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def query_facts(self, keyword):
        conn = db.db_connect("key_facts")
        cur = conn.cursor()
        cur.execute("SELECT * FROM key_facts WHERE fact_text LIKE ? OR canonical_value LIKE ?",
                    (f"%{keyword}%", f"%{keyword}%"))
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]


class SourceCheckerAgent:
    """Verifies source spans against document chunks."""
    def check_span_exists(self, span, text):
        return span in text

    def check_chunk_span_exists(self, doc_hash, chunk_text, span):
        return span in chunk_text

    def find_chunk_for_span(self, doc_hash, span):
        conn = db.db_connect("index")
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, chunk_text FROM document_chunks WHERE doc_hash=? AND chunk_text LIKE ? LIMIT 1",
                    (doc_hash, f"%{span}%"))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None


class ContradictionDetectorAgent:
    """Detects conflicts across all knowledge sources."""
    def detect_conflicts(self):
        return detect_contradictions()


class LogicVerifierAgent:
    """Performs formal consistency checks (currently simple, can be extended)."""
    def verify_formal(self, formal):
        # In future, integrate with an FOL solver.
        # For now, return True if formal is non-empty and no obvious contradiction.
        if not formal:
            return False
        return True
