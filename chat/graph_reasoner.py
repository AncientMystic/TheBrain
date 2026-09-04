
import numpy as np
import config
from core import db
from core.embeddings import get_embedding
from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance

def filter_by_hyperbolic_radius(facts, query_emb, radius=None):
    """Keep facts within a dynamic hyperbolic radius. If radius is None, compute it."""
    if not facts or query_emb is None:
        return facts
    candidate_embs = []
    fact_embs_pairs = []
    for fact in facts:
        text = fact.get("fact_text", "")
        if not text:
            fact_embs_pairs.append((fact, None))
            candidate_embs.append(None)
            continue
        emb = get_embedding(text)
        if emb is None:
            fact_embs_pairs.append((fact, None))
            candidate_embs.append(None)
            continue
        h_emb = ensure_hyperbolic(emb, space='hyperbolic')
        candidate_embs.append(h_emb)
        fact_embs_pairs.append((fact, h_emb))
    valid_embs = [e for e in candidate_embs if e is not None]
    if radius is None:
        from core.dynamic_hyperbolic import dynamic_radius
        radius = dynamic_radius(query_emb, valid_embs,
                                k=getattr(config, "DYNAMIC_RADIUS_K", 10),
                                scale=getattr(config, "DYNAMIC_RADIUS_SCALE", 1.2))
    filtered = []
    for fact, h_emb in fact_embs_pairs:
        if h_emb is not None:
            d = hyperbolic_distance(query_emb, h_emb)
            if d <= radius:
                filtered.append(fact)
        else:
            filtered.append(fact)
    return filtered

def _get_global_node_id(name):
    """Return global_node_id for a canonical name or alias."""
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT global_node_id FROM global_nodes
        WHERE canonical_name = ? OR EXISTS (
            SELECT 1 FROM json_each(global_nodes.aliases_json) WHERE value = ?
        )
        LIMIT 1
    """, (name, name))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def _bfs_paths(start_ids, target_ids, max_depth=3, min_conf=0.6):
    """Find shortest paths from any start to any target using global_edges.
       Returns list of lists: each path is [node_id, relation_type, node_id, ...]."""
    if not start_ids or not target_ids:
        return []
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    # Load all edges once (could be large, but for now okay)
    cur.execute("SELECT source_node_id, target_node_id, relation_type, confidence FROM global_edges WHERE confidence >= ?", (min_conf,))
    edges = cur.fetchall()
    conn.close()

    # Build adjacency list
    adj = {}
    for edge in edges:
        src = edge["source_node_id"]
        tgt = edge["target_node_id"]
        rel = edge["relation_type"]
        conf = edge["confidence"]
        adj.setdefault(src, []).append((tgt, rel, conf))
        adj.setdefault(tgt, []).append((src, rel, conf))  # undirected for path search

    # BFS from each start node
    from collections import deque
    start_set = set(start_ids)
    target_set = set(target_ids)
    found_paths = []
    for start in start_set:
        if start in target_set:
            found_paths.append([start])
            continue
        visited = {start: [start]}
        queue = deque([start])
        while queue and len(found_paths) < 10:
            node = queue.popleft()
            if node in target_set:
                # Reconstruct path
                path = []
                curr = node
                while curr != start:
                    path.append(curr)
                    curr = visited[curr][0]  # parent
                path.append(start)
                path.reverse()
                # Insert relation types by looking up edges later; for now just nodes
                found_paths.append(path)
                break
            for neighbor, rel, conf in adj.get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = [node, rel]
                    queue.append(neighbor)
    return found_paths

def build_reasoning_paths(query_entities, candidate_entities, max_depth=3, min_conf=0.6):
    """Find graph paths between query entities and candidate entities."""
    query_ids = [_get_global_node_id(e) for e in query_entities]
    candidate_ids = [_get_global_node_id(e) for e in candidate_entities]
    query_ids = [i for i in query_ids if i is not None]
    candidate_ids = [i for i in candidate_ids if i is not None]
    if not query_ids or not candidate_ids:
        return []
    return _bfs_paths(query_ids, candidate_ids, max_depth, min_conf)

def format_graph_context(paths, facts):
    """Format paths and facts into a structured context."""
    lines = []
    if paths:
        lines.append("[Verified graph paths:]")
        for path in paths:
            # Convert node ids to names (optional, but we can skip for now)
            lines.append(" -> ".join(str(n) for n in path))
    lines.append("[Selected facts:]")
    for fact in facts:
        text = fact.get("fact_text", "")
        source = fact.get("doc_name", fact.get("doc_hash", "unknown"))
        lines.append(f"- {text} (source: {source})")
    return "\n".join(lines)

def prepare_reasoning_context(query, facts, active_entities=None):
    """Main entry: filter facts by hyperbolic radius and find graph paths."""
    if not facts:
        return ""
    # Get query embedding (already hyperbolic by default)
    q_emb = get_embedding(query)
    if q_emb is None:
        return ""
    q_h = ensure_hyperbolic(q_emb, space='hyperbolic')

    # Filter by radius
    radius = getattr(config, "HYPERBOLIC_FILTER_RADIUS", 1.0)
    filtered = filter_by_hyperbolic_radius(facts, q_h, radius=None)
    if not filtered:
        return ""

    # Extract candidate entities from facts (use canonical_value or simple tokens)
    candidate_entities = set()
    for fact in filtered:
        val = fact.get("canonical_value") or ""
        if val:
            candidate_entities.add(val)
    # Extract query entities (simplified: use all tokens longer than 4)
    from core.text_utils import tokenize
    query_entities = [t for t in tokenize(query) if len(t) > 4]
    if active_entities:
        query_entities.extend([e for e in active_entities if e])

    paths = build_reasoning_paths(query_entities, candidate_entities,
                                  max_depth=getattr(config, "MIN_PATH_DEPTH", 3),
                                  min_conf=getattr(config, "MIN_PATH_CONFIDENCE", 0.6))
    return format_graph_context(paths, filtered)
