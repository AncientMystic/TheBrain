import json
import sqlite3
import numpy as np

import config
from core import db


def _safe_str(value):
    """Convert any value to string, or return empty string for None/dict/list."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return ""
from core.embeddings import get_embeddings_batch
from graph.node_canonicalizer import find_matching_global_node, normalize_name
import logging
logger = logging.getLogger(__name__)


_graph_cache = {
    'lock': None,
    "existing_nodes": None,
    "exact_match_map": None,
    "existing_emb_matrix": None,
    "existing_emb_ids": None,
}


def _get_graph_state(conn, cur):
    if _graph_cache["existing_nodes"] is None:
        cur.execute("SELECT global_node_id, canonical_name, node_type, aliases_json, embedding FROM global_nodes")
        nodes = [dict(row) for row in cur.fetchall()]
        if len(nodes) > getattr(config, "EXTERNAL_GRAPH_CACHE_MAX_NODES", 100000):
            print(f"  (External graph has {len(nodes)} nodes; disabling embedding cache to save RAM)")
            _graph_cache["existing_emb_matrix"] = None
            _graph_cache["existing_emb_ids"] = []
        else:
            existing_embeddings = []
            existing_emb_ids = []
            type_embeddings = {}
            for node in nodes:
                if node.get("embedding") is not None:
                    emb = np.frombuffer(node["embedding"], dtype=np.float32)
                    existing_embeddings.append(emb)
                    existing_emb_ids.append(node["global_node_id"])
                    ntype = node["node_type"]
                    if ntype not in type_embeddings:
                        type_embeddings[ntype] = {"matrix": [], "ids": []}
                    type_embeddings[ntype]["matrix"].append(emb)
                    type_embeddings[ntype]["ids"].append(node["global_node_id"])
            _graph_cache["existing_emb_matrix"] = np.stack(existing_embeddings) if existing_embeddings else None
            _graph_cache["existing_emb_ids"] = existing_emb_ids
            # Convert type lists to matrices
            for ntype in type_embeddings:
                if type_embeddings[ntype]["matrix"]:
                    type_embeddings[ntype]["matrix"] = np.stack(type_embeddings[ntype]["matrix"])
                else:
                    type_embeddings[ntype]["matrix"] = None
            _graph_cache["type_embeddings"] = type_embeddings
        _graph_cache["existing_nodes"] = nodes
        _graph_cache["exact_match_map"] = {}
        for node in _graph_cache["existing_nodes"]:
            norm_name = normalize_name(node["canonical_name"])
            key = (node["node_type"], norm_name)
            _graph_cache["exact_match_map"][key] = node["global_node_id"]
    return (
        _graph_cache["existing_nodes"],
        _graph_cache["exact_match_map"],
        _graph_cache["existing_emb_matrix"],
        _graph_cache["existing_emb_ids"],
    )

def _add_node_to_cache(node):
    import threading
    if _graph_cache.get("lock") is None:
        _graph_cache["lock"] = threading.Lock()
    with _graph_cache["lock"]:
        if _graph_cache["existing_nodes"] is None:
            return
        _graph_cache["existing_nodes"].append(node)
        norm_name = normalize_name(node["canonical_name"])
        key = (node["node_type"], norm_name)
        _graph_cache["exact_match_map"][key] = node["global_node_id"]
        if _graph_cache["existing_emb_matrix"] is not None and node.get("embedding") is not None:
            emb = np.frombuffer(node["embedding"], dtype=np.float32).reshape(1, -1)
            if _graph_cache["existing_emb_matrix"].shape[0] == 0:
                _graph_cache["existing_emb_matrix"] = emb
            else:
                _graph_cache["existing_emb_matrix"] = np.vstack([_graph_cache["existing_emb_matrix"], emb])
            _graph_cache["existing_emb_ids"].append(node["global_node_id"])


def build_external_graph(doc_hash: str, extracted_data: dict, chunk_map: dict) -> None:
    """
    Upsert global nodes and edges from a document's extracted data.
    Batches all embedding calls to reduce latency.
    Uses cached graph state to avoid reloading all nodes per document.
    """
    conn = db.db_connect("external_graph")
    cur = conn.cursor()

    existing_nodes, exact_match_map, existing_emb_matrix, existing_emb_ids = _get_graph_state(conn, cur)
    cache_dirty = False

    # ---- Prepare all candidate names for embedding ----
    all_names = []
    name_type_pairs = []
    for fact in extracted_data.get("facts", []):
        name = _safe_str(fact.get("fact_text"))
        if name:
            all_names.append(name)
            name_type_pairs.append(("FACT", name, fact))
    for ent in extracted_data.get("entities", []):
        name = _safe_str(ent.get("entity_name"))
        if name:
            all_names.append(name)
            name_type_pairs.append((_safe_str(ent.get("entity_type", "ENTITY")), name, ent))
    for person in extracted_data.get("people", []):
        name = _safe_str(person.get("person_name"))
        if name:
            all_names.append(name)
            name_type_pairs.append(("PERSON", name, person))
    for loc in extracted_data.get("locations", []):
        name = _safe_str(loc.get("location_name"))
        if name:
            all_names.append(name)
            name_type_pairs.append(("LOCATION", name, loc))
    for date in extracted_data.get("dates", []):
        name = _safe_str(date.get("date_text"))
        if name:
            all_names.append(name)
            name_type_pairs.append(("DATE", name, date))
    for event in extracted_data.get("events", []):
        name = _safe_str(event.get("event_name"))
        if name:
            all_names.append(name)
            name_type_pairs.append(("EVENT", name, event))
    for disc in extracted_data.get("discoveries", []):
        name = _safe_str(disc.get("discovery_name"))
        if name:
            all_names.append(name)
            name_type_pairs.append(("DISCOVERY", name, disc))
    for gem in extracted_data.get("gems", []):
        name = _safe_str(gem.get("gem_text"))
        if name:
            all_names.append(name)
            name_type_pairs.append(("GEM", name, gem))

    for rel in extracted_data.get("relationships", []):
        src = _safe_str(rel.get("source_node"))
        tgt = _safe_str(rel.get("target_node"))
        if src:
            all_names.append(src)
        if tgt:
            all_names.append(tgt)

    unique_names = list(dict.fromkeys(all_names))

    print(f"  (Batching embeddings for {len(unique_names)} unique names...)")
    embeddings = get_embeddings_batch(unique_names, batch_size=config.EMBEDDING_BATCH_SIZE, space='hyperbolic')
    name_to_embedding = {name: emb for name, emb in zip(unique_names, embeddings) if emb is not None}

    # Precompute embedding matrix for existing global nodes for fast fuzzy matching
    existing_embeddings = []
    existing_emb_ids = []
    for node in existing_nodes:
        if node.get("embedding") is not None:
            existing_embeddings.append(np.frombuffer(node["embedding"], dtype=np.float32))
            existing_emb_ids.append(node["global_node_id"])
    if existing_embeddings:
        existing_emb_matrix = np.stack(existing_embeddings)
    else:
        existing_emb_matrix = None

    local_to_global = {}

    def get_or_create_global_node(node_type, name, attributes=None):
        nonlocal cache_dirty
        name = _safe_str(name)
        if not name:
            return None
        key = (node_type, normalize_name(name))
        if key in local_to_global:
            return local_to_global[key]
        # Fast exact match using in-memory map
        norm_name = normalize_name(name)
        exact_key = (node_type, norm_name)
        if exact_key in exact_match_map:
            local_to_global[key] = exact_match_map[exact_key]
            # Update aliases if new name not in aliases
            cur.execute("SELECT aliases_json FROM global_nodes WHERE global_node_id=?", (exact_match_map[exact_key],))
            row = cur.fetchone()
            aliases = []
            if row and row[0]:
                try:
                    aliases = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    aliases = []
            if name not in aliases:
                aliases.append(name)
                cur.execute("UPDATE global_nodes SET aliases_json=? WHERE global_node_id=?", (json.dumps(aliases), exact_match_map[exact_key]))
            return exact_match_map[exact_key]

        name_emb = name_to_embedding.get(name)
        match_id = find_matching_global_node(name, node_type, existing_nodes, name_embedding=name_emb,
                                                     existing_emb_matrix=existing_emb_matrix, existing_emb_ids=existing_emb_ids)
        if match_id is not None:
            local_to_global[key] = match_id
            cur.execute("SELECT aliases_json FROM global_nodes WHERE global_node_id=?", (match_id,))
            row = cur.fetchone()
            aliases = []
            if row and row[0]:
                try:
                    aliases = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    aliases = []
            if name not in aliases:
                aliases.append(name)
                cur.execute("UPDATE global_nodes SET aliases_json=? WHERE global_node_id=?", (json.dumps(aliases), match_id))
            return match_id

        emb_blob = None
        if name_emb is not None:
            emb_blob = sqlite3.Binary(np.array(name_emb, dtype=np.float32).tobytes())
        attrs_json = json.dumps(attributes or {})
        cur.execute("""
            INSERT INTO global_nodes (canonical_name, node_type, aliases_json, attributes_json, embedding, embedding_space)
            VALUES (?, ?, ?, ?, ?, 'hyperbolic')
        """, (_safe_str(name), _safe_str(node_type), json.dumps([_safe_str(name)]), attrs_json, emb_blob))
        new_id = cur.lastrowid
        # Incrementally update cache
        node_dict = {
            "global_node_id": new_id,
            "canonical_name": _safe_str(name),
            "node_type": _safe_str(node_type),
            "aliases_json": json.dumps([_safe_str(name)]),
            "embedding": emb_blob,
            "embedding_space": "hyperbolic",
        }
        _add_node_to_cache(node_dict)
        cache_dirty = True
        local_to_global[key] = new_id
        return new_id

    # Process all name_type_pairs to create global nodes and cross_doc_links
    for node_type, name, attributes in name_type_pairs:
        gid = get_or_create_global_node(node_type, name, attributes)
        cur.execute("""
            INSERT OR IGNORE INTO cross_doc_links (doc_hash, global_node_id, weight, relation_type)
            VALUES (?, ?, ?, 'mentions')
        """, (doc_hash, gid, 1.0))

    # Normalize relation_type values to strings
    for rel in extracted_data.get("relationships", []):
        if isinstance(rel, dict):
            rel["relation_type"] = _safe_str(rel.get("relation_type", "related"))
    # Handle relationships (endpoints as generic nodes)
    for rel in extracted_data.get("relationships", []):
        src_name = rel.get("source_node")
        tgt_name = rel.get("target_node")
        if not src_name or not tgt_name:
            continue
        src_gid = get_or_create_global_node("OTHER", src_name)
        tgt_gid = get_or_create_global_node("OTHER", tgt_name)

        # Check if edge already exists
        cur.execute("""
            SELECT edge_id FROM global_edges
            WHERE source_node_id=? AND target_node_id=? AND relation_type=?
        """, (src_gid, tgt_gid, rel.get("relation_type", "related")))
        row = cur.fetchone()
        if row:
            cur.execute("""
                UPDATE global_edges
                SET weight = weight + ?,
                    occurrence_count = occurrence_count + 1
                WHERE edge_id=?
            """, (1.0, row[0]))
        else:
            cur.execute("""
                INSERT INTO global_edges (source_node_id, target_node_id, relation_type, weight, doc_hash, source_span, confidence, occurrence_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (src_gid, tgt_gid, _safe_str(rel.get("relation_type", "related")), 1.0,
                  doc_hash, _safe_str(rel.get("evidence_span")), rel.get("confidence", 0.0)))

    # Keyword-topic edges and co-occurrence
    conn_index = db.db_connect("index")
    cur_index = conn_index.cursor()
    cur_index.execute("SELECT filename FROM documents WHERE file_hash=?", (doc_hash,))
    doc_row = cur_index.fetchone()
    doc_name = doc_row[0] if doc_row else "unknown"
    conn_index.close()

    keywords = set()
    for ent in extracted_data.get("entities", []):
        kw = normalize_name(_safe_str(ent.get("entity_name")))
        if kw:
            keywords.add(kw)
            if getattr(config, "ENABLE_TEMPORAL_KEYWORDS", False):
                cur.execute("""
                    INSERT INTO keyword_topic_edges (keyword, topic, weight, first_seen, last_seen)
                    VALUES (?, ?, 1.0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(keyword, topic) DO UPDATE SET
                        weight = weight + 1,
                        last_seen = CURRENT_TIMESTAMP
                """, (kw, doc_name))
            else:
                cur.execute("""
                    INSERT INTO keyword_topic_edges (keyword, topic, weight)
                    VALUES (?, ?, ?)
                    ON CONFLICT(keyword, topic) DO UPDATE SET weight = weight + 1
                """, (_safe_str(kw), _safe_str(doc_name), 1.0))

    keyword_list = list(keywords)
    for i in range(len(keyword_list)):
        for j in range(i+1, len(keyword_list)):
            kw_a, kw_b = sorted([keyword_list[i], keyword_list[j]])
            cur.execute("""
                INSERT INTO keyword_cooccurrence (kw_a, kw_b, weight, doc_hashes)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(kw_a, kw_b) DO UPDATE SET weight = weight + 1,
                    doc_hashes = doc_hashes || ',' || excluded.doc_hashes
            """, (kw_a, kw_b, doc_hash))

    if getattr(config, "ENABLE_GRAPH_KEYWORD_PAGERANK", False):
        try:
            import networkx as nx
            G = nx.Graph()
            for kw_a, kw_b, weight, _ in cur.execute("SELECT kw_a, kw_b, weight, doc_hashes FROM keyword_cooccurrence"):
                G.add_edge(kw_a, kw_b, weight=weight)
            if G.number_of_nodes() > 0:
                pagerank = nx.pagerank(G, weight='weight')
                top_keywords = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
                for kw, score in top_keywords:
                    cur.execute("""
                        INSERT OR REPLACE INTO keyword_topic_edges (keyword, topic, weight)
                        VALUES (?, ?, ?)
                    """, (kw, "__pagerank_top__", score))
        except ImportError:
            print("  (networkx not installed; skipping PageRank)")
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (PageRank error: {e})")

    if getattr(config, "ENABLE_COMMUNITY_DETECTION", False):
        try:
            import networkx as nx
            G = nx.Graph()
            for kw_a, kw_b, weight, _ in cur.execute("SELECT kw_a, kw_b, weight, doc_hashes FROM keyword_cooccurrence"):
                G.add_edge(kw_a, kw_b, weight=weight)
            if G.number_of_nodes() > 0:
                communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
                for cid, community in enumerate(communities):
                    keywords_json = json.dumps(list(community))
                    cur.execute("""
                        INSERT INTO keyword_clusters (cluster_name, keywords_json, size)
                        VALUES (?, ?, ?)
                    """, (f"community_{cid}", keywords_json, len(community)))
        except ImportError:
            print("  (networkx not installed; skipping community detection)")
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Community detection error: {e})")

    conn.commit()
    conn.close()
    # No need to invalidate entire cache; we've updated incrementally.
    # Leave cache_dirty for potential future use.
    print("  (Graph building complete)")
