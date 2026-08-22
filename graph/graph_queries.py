
import time
_keyword_cache = {}
_keyword_cache_ttl = {}

def _cached_get_facts(key, ttl=60):
    now = time.time()
    if key in _keyword_cache and now - _keyword_cache_ttl.get(key, 0) < ttl:
        return _keyword_cache[key]
    return None

def _cache_facts(key, facts, ttl=60):
    _keyword_cache[key] = facts
    _keyword_cache_ttl[key] = time.time()
from core import db
import config

def get_related_keywords(keyword, min_weight=0.5):
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT kw_b, weight FROM keyword_cooccurrence WHERE kw_a=? AND weight>=?
        UNION ALL
        SELECT kw_a, weight FROM keyword_cooccurrence WHERE kw_b=? AND weight>=?
    """, (keyword, min_weight, keyword, min_weight))
    rows = cur.fetchall()
    conn.close()
    return [(row[0], row[1]) for row in rows]


def get_facts_by_keyword(keyword, limit=50):
    """
    Retrieve facts matching a keyword using FTS if available, otherwise LIKE.
    Uses a small in-memory cache for repeated lookups.
    """
    cache_key = f"facts:{keyword}:{limit}"
    cached = _cached_get_facts(cache_key)
    if cached is not None:
        return cached

    if not config.FTS_ENABLED:
        # Fallback to LIKE
        conn = db.db_connect("key_facts")
        cur = conn.cursor()
        cur.execute("""
            SELECT f.fact_id, f.doc_hash, f.doc_name, f.fact_type, f.fact_text,
                   f.canonical_value, f.source_span, f.confidence, fs.chunk_id
            FROM key_facts f
            LEFT JOIN fact_sources fs ON f.fact_id = fs.fact_id
            WHERE f.fact_text LIKE ? OR f.canonical_value LIKE ?
            ORDER BY f.confidence DESC
            LIMIT ?
        """, (f"%{keyword}%", f"%{keyword}%", limit))
        rows = cur.fetchall()
        conn.close()
        result = [dict(row) for row in rows]
        _cache_facts(cache_key, result)
        return result

    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    # Try FTS first
    try:
        cur.execute("""
            SELECT f.fact_id, f.doc_hash, f.doc_name, f.fact_type, f.fact_text,
                   f.canonical_value, f.source_span, f.confidence, fs.chunk_id
            FROM key_facts_fts
            JOIN key_facts f ON key_facts_fts.rowid = f.fact_id
            LEFT JOIN fact_sources fs ON f.fact_id = fs.fact_id
            WHERE key_facts_fts MATCH ?
            ORDER BY f.confidence DESC
            LIMIT ?
        """, (keyword, limit))
        rows = cur.fetchall()
        if rows:
            conn.close()
            result = [dict(row) for row in rows]
            _cache_facts(cache_key, result)
            return result
    except Exception:
        pass  # FTS not available or query failed

    # Fallback to LIKE
    cur.execute("""
        SELECT f.fact_id, f.doc_hash, f.doc_name, f.fact_type, f.fact_text,
               f.canonical_value, f.source_span, f.confidence, fs.chunk_id
        FROM key_facts f
        LEFT JOIN fact_sources fs ON f.fact_id = fs.fact_id
        WHERE f.fact_text LIKE ? OR f.canonical_value LIKE ?
        ORDER BY f.confidence DESC
        LIMIT ?
    """, (f"%{keyword}%", f"%{keyword}%", limit))
    rows = cur.fetchall()
    conn.close()
    result = [dict(row) for row in rows]
    _cache_facts(cache_key, result)
    return result


def get_global_node_edges(global_node_id):
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT e.edge_id, e.source_node_id, e.target_node_id, e.relation_type, e.weight, e.doc_hash, e.source_span, e.confidence
        FROM global_edges e
        WHERE e.source_node_id=? OR e.target_node_id=?
        ORDER BY e.weight DESC
    """, (global_node_id, global_node_id))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]
