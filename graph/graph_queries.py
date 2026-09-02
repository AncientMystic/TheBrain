
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
from fuzzywuzzy import fuzz
import logging
logger = logging.getLogger(__name__)

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


def _keyword_variants(keyword):
    """Return a list of simple variants for a keyword."""
    variants = {keyword, keyword.lower(), keyword.title()}
    if keyword.endswith("s") and len(keyword) > 3:
        variants.add(keyword[:-1])
    if keyword.endswith("ies") and len(keyword) > 4:
        variants.add(keyword[:-3] + "y")
    return [v for v in variants if v]


def get_facts_by_keyword(keyword, limit=50):
    """
    Retrieve facts matching a keyword using FTS, LIKE, and fuzzy ranking.
    Returns fact dicts sorted by hybrid confidence + fuzzy relevance.
    """
    if not keyword:
        return []

    cache_key = f"facts:{keyword}:{limit}"
    cached = _cached_get_facts(cache_key)
    if cached is not None:
        return cached

    variants = _keyword_variants(keyword)

    collected = {}

    # FTS prefix search
    if config.FTS_ENABLED:
        conn = db.db_connect("key_facts")
        cur = conn.cursor()
        for v in variants:
            fts_query = v.replace('"', '""') + "*"
            try:
                cur.execute("""
                    SELECT f.fact_id, f.doc_hash, f.doc_name, f.fact_type, f.fact_text,
                           f.canonical_value, f.source_span, f.confidence, fs.chunk_id
                    FROM key_facts_fts
                    JOIN key_facts f ON key_facts_fts.rowid = f.fact_id
                    LEFT JOIN fact_sources fs ON f.fact_id = fs.fact_id
                    WHERE key_facts_fts MATCH ?
                    LIMIT ?
                """, (fts_query, limit * 3))
                for row in cur.fetchall():
                    d = dict(row)
                    collected[d["fact_id"]] = d
            except Exception:
                logger.warning("Unexpected exception occurred", exc_info=True)
                pass
        conn.close()

    # LIKE fallback / supplement
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    likes = []
    params = []
    for v in variants:
        likes.append("(f.fact_text LIKE ? OR f.canonical_value LIKE ?)")
        params.extend([f"%{v}%", f"%{v}%"])
    like_sql = " OR ".join(likes) if likes else "1=0"
    cur.execute(f"""
        SELECT f.fact_id, f.doc_hash, f.doc_name, f.fact_type, f.fact_text,
               f.canonical_value, f.source_span, f.confidence, fs.chunk_id
        FROM key_facts f
        LEFT JOIN fact_sources fs ON f.fact_id = fs.fact_id
        WHERE {like_sql}
        LIMIT ?
    """, (*params, limit * 5))
    for row in cur.fetchall():
        d = dict(row)
        collected[d["fact_id"]] = d
    conn.close()

    # Fuzzy rank
    results = []
    kw = keyword.lower()
    for d in collected.values():
        text = (d.get("fact_text") or "").lower()
        canonical = (d.get("canonical_value") or "").lower()
        best = max(
            fuzz.token_set_ratio(kw, text),
            fuzz.partial_token_set_ratio(kw, text),
            fuzz.token_set_ratio(kw, canonical),
            fuzz.partial_token_set_ratio(kw, canonical),
        )
        if kw in text or kw in canonical:
            best = max(best, 100)

        rank_score = (d.get("confidence") or 0.0) * 0.6 + (best / 100.0) * 0.4
        d["_fuzzy_score"] = best
        d["_rank_score"] = rank_score
        results.append(d)

    results.sort(key=lambda x: x["_rank_score"], reverse=True)
    result = results[:limit]
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
