"""
Read-only graph APIs for Deep Graph mindmap (no writes, no LLM calls).

- nodes: top-degree global nodes (cap, deterministic order)
- expand: 1-hop neighbors of a node id with edge labels
- facts: facts linked to a keyword/canonical name via existing FTS path
All paginated/capped (never unbounded), generic, reuse existing queries.
"""
import config


def register_graph_routes(app, require_auth):
    from fastapi import Depends

    @app.get("/api/graph/nodes", dependencies=[Depends(require_auth)])
    async def graph_nodes(limit: int = 100, query: str = ""):
        from core import db
        limit = max(1, min(int(limit or 100), 300))
        conn = db.db_connect("external_graph")
        try:
            cur = conn.cursor()
            if query:
                like = f"%{query}%"
                cur.execute("""SELECT global_node_id, canonical_name, node_type FROM global_nodes
                               WHERE canonical_name LIKE ? ORDER BY global_node_id LIMIT ?""", (like, limit))
            else:
                # Top-degree nodes first (most connected = most informative overview)
                cur.execute("""SELECT n.global_node_id, n.canonical_name, n.node_type,
                                      COUNT(e.source_node_id) + COUNT(e.target_node_id) AS deg
                               FROM global_nodes n LEFT JOIN global_edges e
                                 ON e.source_node_id=n.global_node_id OR e.target_node_id=n.global_node_id
                               GROUP BY n.global_node_id ORDER BY deg DESC LIMIT ?""", (limit,))
            rows = cur.fetchall()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        nodes = [{"id": r["global_node_id"], "label": r["canonical_name"], "type": r["node_type"]} for r in rows]
        # Edges among returned nodes (single IN query, capped)
        edges = []
        if nodes:
            ids = [n["id"] for n in nodes]
            conn2 = db.db_connect("external_graph")
            try:
                cur2 = conn2.cursor()
                for s in range(0, len(ids), 200):
                    ch = ids[s:s + 200]
                    ph = ",".join("?" for _ in ch)
                    cur2.execute(f"""SELECT source_node_id, target_node_id, relation_type FROM global_edges
                                     WHERE source_node_id IN ({ph}) AND target_node_id IN ({ph}) LIMIT 500""",
                                 (*ch, *ch))
                    for r in cur2.fetchall():
                        edges.append({"from": r["source_node_id"], "to": r["target_node_id"], "label": r["relation_type"]})
            finally:
                try:
                    conn2.close()
                except Exception:
                    pass
        return {"nodes": nodes, "edges": edges}

    @app.get("/api/graph/expand/{node_id}", dependencies=[Depends(require_auth)])
    async def graph_expand(node_id: int):
        from graph.graph_queries import get_global_node_edges
        from core import db
        edges = get_global_node_edges(node_id)
        # Batch neighbor names (no N+1)
        other_ids = list({(e["source_node_id"] if e["source_node_id"] != node_id else e["target_node_id"]) for e in edges})
        names = {}
        if other_ids:
            conn = db.db_connect("external_graph")
            try:
                cur = conn.cursor()
                for s in range(0, len(other_ids), 200):
                    ch = other_ids[s:s + 200]
                    ph = ",".join("?" for _ in ch)
                    cur.execute(f"SELECT global_node_id, canonical_name, node_type FROM global_nodes WHERE global_node_id IN ({ph})", ch)
                    for r in cur.fetchall():
                        names[r["global_node_id"]] = {"label": r["canonical_name"], "type": r["node_type"]}
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        nodes = [{"id": oid, **names.get(oid, {"label": str(oid), "type": "unknown"})} for oid in other_ids]
        edges_out = [{"from": e["source_node_id"], "to": e["target_node_id"], "label": e.get("relation_type", "")} for e in edges]
        return {"nodes": nodes, "edges": edges_out}

    @app.get("/api/graph/facts", dependencies=[Depends(require_auth)])
    async def graph_facts(keyword: str, limit: int = 20):
        from graph.graph_queries import get_facts_by_keyword
        limit = max(1, min(int(limit or 20), 50))
        facts = get_facts_by_keyword(keyword)[:limit]
        out = []
        for f in facts:
            out.append({"fact_id": f.get("fact_id"), "fact_text": f.get("fact_text", "")[:300],
                        "confidence": f.get("confidence", 0), "doc": f.get("doc_name", f.get("doc_hash", ""))})
        return {"facts": out}

    return app
