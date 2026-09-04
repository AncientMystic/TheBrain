import numpy as np
import config
from core import db
from core.embeddings import get_embeddings_batch
import logging
logger = logging.getLogger(__name__)


def hyperbolic_similarity(a, b):
    from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance
    ah = ensure_hyperbolic(a, space='hyperbolic')
    bh = ensure_hyperbolic(b, space='hyperbolic')
    try:
        d = float(hyperbolic_distance(ah, bh))
    except Exception:
        return 0.0
    return 1.0 / (1.0 + d)


def cosine_similarity(a, b):
    # Backward-compat alias: now geometry-correct via hyperbolic distance.
    return hyperbolic_similarity(a, b)


def semantic_audit_keyword_edges():
    """
    Remove keyword-topic edges where keyword and topic are semantically unrelated.
    Uses batched embeddings for speed.
    """
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("SELECT keyword, topic, weight FROM keyword_topic_edges")
    rows = cur.fetchall()
    if not rows:
        conn.close()
        return []

    # Collect unique keywords and topics
    topics = sorted({t for _, t, _ in rows})
    keywords = sorted({k for k, _, _ in rows if len(k) > 2})

    print(f"  (Semantic audit: batching embeddings for {len(topics)} topics and {len(keywords)} keywords...)")

    # Batch embeddings
    topic_emb_list = get_embeddings_batch(topics, batch_size=config.EMBEDDING_BATCH_SIZE)
    keyword_emb_list = get_embeddings_batch(keywords, batch_size=config.EMBEDDING_BATCH_SIZE)

    topic_emb = {t: emb for t, emb in zip(topics, topic_emb_list) if emb is not None}
    keyword_emb = {kw: emb for kw, emb in zip(keywords, keyword_emb_list) if emb is not None}

    candidates = []

    for kw, topic, weight in rows:
        if len(kw) <= 2:
            candidates.append((kw, topic))
            continue

        kw_emb = keyword_emb.get(kw)
        t_emb = topic_emb.get(topic)
        if kw_emb is None or t_emb is None:
            continue

        sim = cosine_similarity(kw_emb, t_emb)
        if sim < config.SEMANTIC_SIM_THRESHOLD:
            candidates.append((kw, topic))

    # Remove candidates
    for kw, topic in candidates:
        cur.execute("DELETE FROM keyword_topic_edges WHERE keyword=? AND topic=?", (kw, topic))

    conn.commit()
    conn.close()
    print(f"  (Removed {len(candidates)} semantically unrelated keyword-topic edges)")
    return candidates
