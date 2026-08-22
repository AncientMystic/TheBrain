"""
MindMap storage for deep research.
Uses reasoning.db tables research_nodes and research_edges.
"""
import json
from core import db

def init_mindmap_db():
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS research_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            research_id TEXT NOT NULL,
            node_type TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT,
            confidence REAL DEFAULT 0.0,
            metadata_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS research_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            research_id TEXT NOT NULL,
            source_node_id INTEGER REFERENCES research_nodes(id),
            target_node_id INTEGER REFERENCES research_nodes(id),
            relation_type TEXT,
            weight REAL DEFAULT 1.0,
            evidence TEXT,
            confidence REAL DEFAULT 0.0
        )
    """)
    conn.commit()
    conn.close()

def add_research_node(research_id, node_type, name, content="", confidence=0.0, metadata=None):
    init_mindmap_db()
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO research_nodes (research_id, node_type, name, content, confidence, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (research_id, node_type, name, content, confidence, json.dumps(metadata or {})))
    node_id = cur.lastrowid
    conn.commit()
    conn.close()
    return node_id

def add_research_edge(research_id, source_id, target_id, relation_type, evidence=None, confidence=0.0):
    init_mindmap_db()
    conn = db.db_connect("reasoning")
    conn.execute("""
        INSERT INTO research_edges (research_id, source_node_id, target_node_id, relation_type, evidence, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (research_id, source_id, target_id, relation_type, evidence, confidence))
    conn.commit()
    conn.close()

def get_mindmap_text(research_id):
    """Return a simple indented text representation of the mindmap."""
    init_mindmap_db()
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("SELECT id, node_type, name FROM research_nodes WHERE research_id=?", (research_id,))
    nodes = {row["id"]: row for row in cur.fetchall()}
    cur.execute("SELECT source_node_id, target_node_id, relation_type FROM research_edges WHERE research_id=?", (research_id,))
    edges = [(row["source_node_id"], row["target_node_id"], row["relation_type"]) for row in cur.fetchall()]
    conn.close()
    # Build adjacency
    adj = {}
    for src, tgt, rel in edges:
        adj.setdefault(src, []).append((tgt, rel))
    # Traverse from roots (nodes with no incoming edges)
    has_incoming = {tgt for _, tgt, _ in edges}
    roots = [id for id in nodes if id not in has_incoming]
    if not roots:
        roots = list(nodes.keys())[:1]
    lines = []
    visited = set()
    def dfs(node_id, indent=0):
        if node_id in visited:
            return
        visited.add(node_id)
        node = nodes[node_id]
        lines.append("  "*indent + f"- [{node['node_type']}] {node['name']}")
        for child, rel in adj.get(node_id, []):
            lines.append("  "*(indent+1) + f"({rel})")
            dfs(child, indent+2)
    for root in roots:
        dfs(root)
    return "\n".join(lines)


def build_hierarchical_mindmap(research_id, facts, global_edges):
    """Build a richer mindmap using facts and external graph edges."""
    root_id = add_research_node(research_id, "topic", "Research Topic", content="Main topic", confidence=1.0)
    fact_nodes = {}
    entity_nodes = {}
    # Create fact nodes
    for fact in facts[:200]:
        fact_text = fact.get("fact_text", "")
        fact_id = fact.get("fact_id")
        if not fact_id:
            continue
        node_id = add_research_node(research_id, "fact", fact_text[:200], content=fact_text, confidence=fact.get("confidence",0.5))
        fact_nodes[fact_id] = node_id
        add_research_edge(research_id, root_id, node_id, "contains")
    # Create entity nodes and connect to facts via mention (simplified: use canonical_value)
    for fact in facts[:100]:
        canonical = fact.get("canonical_value")
        if not canonical:
            continue
        if canonical not in entity_nodes:
            entity_id = add_research_node(research_id, "entity", canonical, content=canonical, confidence=fact.get("confidence",0.5))
            entity_nodes[canonical] = entity_id
        # Link fact to entity
        if fact.get("fact_id") in fact_nodes:
            add_research_edge(research_id, fact_nodes[fact["fact_id"]], entity_nodes[canonical], "mentions")
    # Connect entities using global_edges
    for edge in global_edges:
        src = edge.get("source_node")
        tgt = edge.get("target_node")
        if src in entity_nodes and tgt in entity_nodes:
            add_research_edge(research_id, entity_nodes[src], entity_nodes[tgt], edge.get("relation_type","related"), confidence=edge.get("confidence",0.5))
