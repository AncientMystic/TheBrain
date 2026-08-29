
import numpy as np
from collections import defaultdict, deque
import config
from core import db
from core.embeddings import get_embedding
from core.hyperbolic import exp_map, log_map, hyperbolic_distance
from core.hyperbolic_clustering import cluster_hyperbolic, select_representatives

def _fact_embedding(fact):
    """Return hyperbolic embedding for a fact, or None."""
    text = fact.get("fact_text", "")
    if not text:
        return None
    emb = get_embedding(text)
    if emb is None:
        return None
    return exp_map(np.array(emb, dtype=np.float32))

def _get_node_id_for_entity(entity):
    """Look up global_node_id by canonical name or alias."""
    if not entity:
        return None
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT global_node_id FROM global_nodes
        WHERE canonical_name = ? OR EXISTS (
            SELECT 1 FROM json_each(global_nodes.aliases_json) WHERE value = ?
        )
        LIMIT 1
    """, (entity, entity))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def _find_connected_components(node_ids, edges):
    """Return connected components from a list of node_ids and edges (list of (src,tgt,rel))."""
    adj = defaultdict(list)
    nodes = set(node_ids)
    for src, tgt, rel in edges:
        if src in nodes and tgt in nodes:
            adj[src].append((tgt, rel))
            adj[tgt].append((src, rel))
    visited = set()
    components = []
    for node in nodes:
        if node in visited:
            continue
        comp_nodes = []
        comp_edges = []
        queue = deque([node])
        visited.add(node)
        while queue:
            cur = queue.popleft()
            comp_nodes.append(cur)
            for neighbor, rel in adj[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                if (cur, neighbor, rel) not in comp_edges and (neighbor, cur, rel) not in comp_edges:
                    comp_edges.append((cur, neighbor, rel))
        components.append((comp_nodes, comp_edges))
    return components

def _get_all_edges_for_nodes(node_ids):
    """Fetch all edges among the given node_ids from global_edges."""
    if not node_ids:
        return []
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in node_ids)
    cur.execute(f"""
        SELECT source_node_id, target_node_id, relation_type
        FROM global_edges
        WHERE source_node_id IN ({placeholders}) AND target_node_id IN ({placeholders})
    """, (*node_ids, *node_ids))
    rows = cur.fetchall()
    conn.close()
    return [(r["source_node_id"], r["target_node_id"], r["relation_type"]) for r in rows]

def organize_facts(facts, query_embedding=None, session_id=None, max_clusters=10, active_entities=None):
    """
    Group facts into semantic clusters and graph-connected components.
    Returns a structured text block for context.
    """
    if not facts:
        return ""

    # 1. Embed facts and map to hyperbolic
    fact_embeddings = []
    valid_facts = []
    for fact in facts:
        emb = _fact_embedding(fact)
        if emb is not None:
            fact_embeddings.append(emb)
            valid_facts.append(fact)

    if not valid_facts:
        # Fallback: return facts as plain list
        lines = []
        for fact in facts:
            text = fact.get("fact_text", "")
            source = fact.get("doc_name", fact.get("doc_hash", "unknown"))
            lines.append(f"- {text} (source: {source})")
        return "
".join(lines)

    # 2. Cluster facts hyperbolically
    n_clusters = min(max_clusters, len(valid_facts))
    clusters = cluster_hyperbolic(fact_embeddings, n_clusters=n_clusters)

    # If active_entities provided, keep facts that mention any active entity or are graph-connected later
    if active_entities:
        # Mark facts that mention active entities
        active_set = set(e.lower() for e in active_entities)
        # We'll use this to weight cluster assignment: facts with active entity go to dedicated group? Simpler: just annotate.
        pass
    # 3. For each cluster, find graph-connected components
    lines = []
    for cluster_idx, cluster in enumerate(clusters, 1):
        if not cluster:
            continue
        cluster_facts = [valid_facts[i] for i in cluster]
        cluster_embs = [fact_embeddings[i] for i in cluster]
        # Collect entity mentions for graph lookup
        node_ids = set()
        for fact in cluster_facts:
            val = fact.get("canonical_value") or fact.get("fact_text") or ""
            # Use first few tokens as possible entity
            from core.text_utils import tokenize
            tokens = tokenize(val)
            for t in tokens[:3]:
                nid = _get_node_id_for_entity(t)
                if nid:
                    node_ids.add(nid)

        if node_ids:
            edges = _get_all_edges_for_nodes(list(node_ids))
            components = _find_connected_components(list(node_ids), edges)
        else:
            components = [(list(node_ids), [])]

        # If multiple components, split facts by component
        if len(components) > 1 and node_ids:
            # Map fact to component by checking which entity appears in which component
            component_facts = [[] for _ in components]
            for fact, emb in zip(cluster_facts, cluster_embs):
                val = fact.get("canonical_value") or fact.get("fact_text") or ""
                tokens = tokenize(val)[:3]
                assigned = False
                for ci, (comp_nodes, _) in enumerate(components):
                    for t in tokens:
                        nid = _get_node_id_for_entity(t)
                        if nid and nid in comp_nodes:
                            component_facts[ci].append((fact, emb))
                            assigned = True
                            break
                    if assigned:
                        break
                if not assigned:
                    component_facts[0].append((fact, emb))  # default to first
        else:
            component_facts = [[(fact, emb) for fact, emb in zip(cluster_facts, cluster_embs)]]

        # Format groups
        for ci, cfacts in enumerate(component_facts):
            if not cfacts:
                continue
            lines.append(f"[Group {cluster_idx}.{ci+1}]")
            # Find a representative title from first fact
            rep_fact = cfacts[0][0]
            rep_text = rep_fact.get("fact_text", "")[:80]
            lines.append(f"Topic: {rep_text} ...")
            for fact, emb in cfacts:
                text = fact.get("fact_text", "")
                source = fact.get("doc_name", fact.get("doc_hash", "unknown"))
                lines.append(f"- {text} (source: {source})")
            # Add graph edges if present
            if node_ids and components:
                _, comp_edges = components[ci] if ci < len(components) else ([], [])
                for src, tgt, rel in comp_edges[:5]:  # limit to avoid clutter
                    # Get names for src/tgt
                    src_name = ""
                    tgt_name = ""
                    # we can skip names for now; just show relation
                    lines.append(f"[Graph edge: {src} -> {rel} -> {tgt}]")
            lines.append("")  # blank line between groups

    if not lines:
        # Fallback: plain list
        for fact in facts:
            text = fact.get("fact_text", "")
            source = fact.get("doc_name", fact.get("doc_hash", "unknown"))
            lines.append(f"- {text} (source: {source})")

    return "
".join(lines)
