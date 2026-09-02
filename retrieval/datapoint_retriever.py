"""
Graph-native datapoint retriever.

Gathers a connected subgraph of small atomic datapoints from
TheBrain's multiple knowledge graphs and ranks them for LLM selection.
"""

import time
import numpy as np
from core import db
from core.embeddings import get_embedding
import config
from core.reranker import get_reranker
from core.local_embedder import get_local_embedder
import logging
logger = logging.getLogger(__name__)

AD_MARKERS = [
    "spotify for podcasters",
    "packed calendar",
    "spotify for podcasters",
    "start today",
    "commercial",
    "packed calendar today",
    "check out spotify",
]


def _tokenize(text):
    from core.text_utils import tokenize
    return set(tokenize(text))


def _phrase_overlap(text, query_tokens):
    """Return 1.0 if most query tokens appear as a consecutive phrase."""
    lowered = text.lower()
    phrase = " ".join(sorted(query_tokens, key=len, reverse=True))
    if phrase and phrase in lowered:
        return 1.0
    # fallback: check any big token sequence
    tokens = lowered.split()
    for i in range(len(tokens)-1):
        pair = " ".join(tokens[i:i+2])
        if pair and pair in lowered:
            return 0.5
    return 0.0


def _score_datapoint(dp, query_tokens, root_distance):
    """Compute weighted relevance score."""
    text = (dp.get("text") or "").lower()
    tokens = _tokenize(text)
    overlap = len(tokens & query_tokens) / max(1, len(query_tokens))
    confidence = dp.get("confidence", 0.5)
    graph_proximity = 1.0 / (1.0 + root_distance)
    # Semantic similarity optional; skipped if no embeddings available.
    weights = getattr(config, "DATAPOINT_SCORE_WEIGHTS", {})
    score = (
        weights.get("query_overlap", 0.35) * overlap +
        weights.get("graph_proximity", 0.25) * graph_proximity +
        weights.get("confidence", 0.15) * confidence
    )
    return score


def retrieve_datapoints(query, max_nodes=None, depth=None, extra_terms=None):
    """
    Hierarchical retrieval:
    1. Find candidate documents by query terms.
    2. Score documents using summaries and titles with reranker.
    3. Retrieve chunks, facts, and summaries from top documents only.
    4. Score datapoints with query overlap, document boost, and reranker.
    """
    if max_nodes is None:
        max_nodes = getattr(config, "MAX_MAP_NODES", 200)
    if depth is None:
        depth = getattr(config, "EXPANSION_DEPTH", 2)

    query_tokens = _tokenize(query)
    if extra_terms:
        query_tokens = query_tokens | _tokenize(" ".join(extra_terms))

    from chat.query_analyzer import analyze_query
    analysis = analyze_query(query)
    root_terms = [t for t in analysis.get("keywords", []) if t]
    if extra_terms:
        root_terms.extend([t for t in extra_terms if t not in root_terms])

    # --------------------------------------------------------------
    # Stage 1: Find candidate documents from titles, chunks, and summaries
    # --------------------------------------------------------------
    candidate_docs = {}

    def add_doc(doc_hash, doc_name):
        if doc_hash and doc_hash not in candidate_docs:
            candidate_docs[doc_hash] = doc_name or doc_hash

    try:
        conn = db.db_connect("index")
        cur = conn.cursor()

        # 1a. Direct document title/filename matches
        for term in root_terms:
            cur.execute("""
                SELECT file_hash, filename, title
                FROM documents
                WHERE filename LIKE ? OR title LIKE ?
                LIMIT 10
            """, (f"%{term}%", f"%{term}%"))
            for doc_hash, filename, title in cur.fetchall():
                add_doc(doc_hash, title or filename or doc_hash)

        # 1b. Documents whose chunks mention the term
        for term in root_terms:
            cur.execute("""
                SELECT DISTINCT dc.doc_hash, d.filename, d.title
                FROM document_chunks dc
                JOIN documents d ON dc.doc_hash = d.file_hash
                WHERE dc.chunk_text LIKE ?
                LIMIT 20
            """, (f"%{term}%",))
            for doc_hash, filename, title in cur.fetchall():
                add_doc(doc_hash, title or filename or doc_hash)

        conn.close()
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"    (Document candidate retrieval error: {e})")

    # 1c. Documents whose summary mentions the term
    try:
        conn_summ = db.db_connect("summaries")
        cur_summ = conn_summ.cursor()
        for term in root_terms:
            cur_summ.execute("""
                SELECT doc_hash, doc_name, summary
                FROM doc_summaries
                WHERE doc_name LIKE ? OR summary LIKE ?
                LIMIT 10
            """, (f"%{term}%", f"%{term}%"))
            for doc_hash, doc_name, summary in cur_summ.fetchall():
                add_doc(doc_hash, doc_name or doc_hash)
        conn_summ.close()
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"    (Summary candidate retrieval error: {e})")

    if not candidate_docs:
        return _direct_chunk_fallback(query, root_terms, query_tokens, max_nodes)

    # --------------------------------------------------------------
    # Stage 2: Score candidate documents using summaries and reranker
    # --------------------------------------------------------------
    doc_scores = []
    reranker = get_reranker()
    for doc_hash, doc_name in candidate_docs.items():
        summary_text = ""
        try:
            conn = db.db_connect("summaries")
            cur = conn.cursor()
            cur.execute("SELECT summary FROM doc_summaries WHERE doc_hash=?", (doc_hash,))
            row = cur.fetchone()
            conn.close()
            if row:
                summary_text = row["summary"] or ""
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            pass
        combined_text = f"{doc_name} {summary_text}"[:1000]
        if reranker.available:
            try:
                score = reranker.score(query, [combined_text])[0]
            except Exception:
                score = 0.0
        else:
            # fallback: token overlap
            overlap = len(_tokenize(combined_text) & query_tokens) / max(1, len(query_tokens))
            score = overlap
        doc_scores.append((score, doc_hash, doc_name, summary_text))

    doc_scores.sort(key=lambda x: x[0], reverse=True)
    top_docs = doc_scores[:3]  # top 3 documents

    # --------------------------------------------------------------
    # Stage 3: Collect datapoints from top documents
    # --------------------------------------------------------------
    datapoints = []
    seen_ids = set()

    for doc_score, doc_hash, doc_name, summary_text in top_docs:
        # Document datapoint
        dp_id = f"document:{doc_hash}"
        if dp_id not in seen_ids:
            seen_ids.add(dp_id)
            datapoints.append({
                "id": dp_id,
                "type": "document",
                "text": doc_name,
                "confidence": 0.95,
                "doc_hash": doc_hash,
                "chunk_id": None,
                "source_span": None,
                "_root_distance": 0,
                "_doc_relevance": doc_score,
            })

        # Summary datapoint
        if summary_text:
            dp_id = f"summary:{doc_hash}"
            if dp_id not in seen_ids:
                seen_ids.add(dp_id)
                datapoints.append({
                    "id": dp_id,
                    "type": "summary",
                    "text": summary_text[:300],
                    "confidence": 0.9,
                    "doc_hash": doc_hash,
                    "chunk_id": None,
                    "source_span": None,
                    "_root_distance": 0,
                    "_doc_relevance": doc_score,
                })

        # Facts from this document
        try:
            conn = db.db_connect("key_facts")
            cur = conn.cursor()
            cur.execute("SELECT fact_id, fact_text, canonical_value, confidence, source_span FROM key_facts WHERE doc_hash=? LIMIT 20", (doc_hash,))
            for row in cur.fetchall():
                dp_id = f"fact:{row['fact_id']}"
                if dp_id not in seen_ids:
                    seen_ids.add(dp_id)
                    datapoints.append({
                        "id": dp_id,
                        "type": "fact",
                        "text": row["fact_text"],
                        "confidence": row["confidence"],
                        "doc_hash": doc_hash,
                        "chunk_id": None,
                        "source_span": row["source_span"],
                        "_root_distance": 0,
                        "_doc_relevance": doc_score,
                    })
            conn.close()
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Fact retrieval error for doc {doc_hash}: {e})")

        # Chunks from this document containing root terms
        try:
            conn = db.db_connect("index")
            cur = conn.cursor()
            likes = []
            params = []
            for term in root_terms:
                likes.append("chunk_text LIKE ?")
                params.append(f"%{term}%")
            like_sql = " OR ".join(likes) if likes else "1=0"
            cur.execute(f"SELECT chunk_id, doc_hash, chunk_text FROM document_chunks WHERE doc_hash=? AND ({like_sql}) LIMIT 15", (doc_hash, *params))
            for cid, dhash, ctext in cur.fetchall():
                dp_id = f"chunk_ref:{cid}"
                if dp_id not in seen_ids:
                    seen_ids.add(dp_id)
                    datapoints.append({
                        "id": dp_id,
                        "type": "chunk_ref",
                        "text": ctext[:250],
                        "confidence": 0.8,
                        "doc_hash": dhash,
                        "chunk_id": cid,
                        "source_span": None,
                        "_root_distance": 0,
                        "_is_chunk": True,
                        "_doc_relevance": doc_score,
                    })
            conn.close()
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Chunk retrieval error for doc {doc_hash}: {e})")

    # --------------------------------------------------------------
    # Stage 4: Score datapoints with the new ranking model
    # --------------------------------------------------------------
    from retrieval.ranking import get_ranker
    ranker = get_ranker()

    # Extract query entities for graph proximity feature
    query_entities = [ent.get('text', '') for ent in analysis.get('entities', [])]

    scores = ranker.batch_score(query, datapoints, query_entities, reranker)
    for dp, score in zip(datapoints, scores):
        dp['score'] = score

    # Sort descending by score
    datapoints.sort(key=lambda x: x.get('score', 0), reverse=True)
    return datapoints[:max_nodes]


def _direct_chunk_fallback(query, root_terms, _query_tokens, max_nodes):
    """Fallback when no documents match: direct chunk search, but include doc info."""
    datapoints = []
    seen_ids = set()

    try:
        conn = db.db_connect("index")
        cur = conn.cursor()
        doc_cache = {}

        for term in root_terms:
            cur.execute(
                "SELECT chunk_id, doc_hash, chunk_text FROM document_chunks WHERE chunk_text LIKE ? LIMIT 20",
                (f"%{term}%",)
            )
            for cid, dhash, ctext in cur.fetchall():
                # Fetch document title and add document datapoint if not seen
                if dhash not in doc_cache:
                    try:
                        conn2 = db.db_connect("index")
                        cur2 = conn2.cursor()
                        cur2.execute("SELECT filename, title FROM documents WHERE file_hash=?", (dhash,))
                        row = cur2.fetchone()
                        conn2.close()
                        doc_name = (row["title"] or row["filename"]) if row else dhash
                    except Exception:
                        doc_name = dhash
                    doc_cache[dhash] = doc_name

                    doc_dp_id = f"document:{dhash}"
                    if doc_dp_id not in seen_ids:
                        seen_ids.add(doc_dp_id)
                        datapoints.append({
                            "id": doc_dp_id,
                            "type": "document",
                            "text": doc_name,
                            "confidence": 0.95,
                            "doc_hash": dhash,
                            "chunk_id": None,
                            "source_span": None,
                            "_root_distance": 0,
                        })

                    # Add summary if available
                    try:
                        conn_summ = db.db_connect("summaries")
                        cur_summ = conn_summ.cursor()
                        cur_summ.execute("SELECT summary FROM doc_summaries WHERE doc_hash=?", (dhash,))
                        srow = cur_summ.fetchone()
                        conn_summ.close()
                        if srow and srow["summary"]:
                            sdp_id = f"summary:{dhash}"
                            if sdp_id not in seen_ids:
                                seen_ids.add(sdp_id)
                                datapoints.append({
                                    "id": sdp_id,
                                    "type": "summary",
                                    "text": srow["summary"][:300],
                                    "confidence": 0.9,
                                    "doc_hash": dhash,
                                    "chunk_id": None,
                                    "source_span": None,
                                    "_root_distance": 0,
                                })
                    except Exception:
                        logger.warning("Unexpected exception occurred", exc_info=True)
                        pass

                dp_id = f"chunk_ref:{cid}"
                if dp_id in seen_ids:
                    continue
                seen_ids.add(dp_id)
                datapoints.append({
                    "id": dp_id,
                    "type": "chunk_ref",
                    "text": ctext[:250],
                    "confidence": 0.7,
                    "doc_hash": dhash,
                    "chunk_id": cid,
                    "source_span": None,
                    "_root_distance": 0,
                    "_is_chunk": True,
                })
        conn.close()
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"    (Direct chunk fallback error: {e})")

    # Score with batched ranker
    from retrieval.ranking import get_ranker
    ranker = get_ranker()
    scores = ranker.batch_score(query, datapoints, [], None)
    for dp, score in zip(datapoints, scores):
        dp["score"] = score
    datapoints.sort(key=lambda x: x.get("score", 0), reverse=True)
    return datapoints[:max_nodes]

def get_chunks_for_datapoints(selected_datapoints):
    """
    Retrieve exact chunk texts for selected datapoints.
    Returns list of (chunk_id, doc_hash, text).
    """
    chunk_ids = set()
    for dp in selected_datapoints:
        cid = dp.get("chunk_id")
        if cid:
            chunk_ids.add(cid)

    if not chunk_ids:
        return []

    conn = db.db_connect("index")
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in chunk_ids)
    cur.execute(f"SELECT chunk_id, doc_hash, chunk_text FROM document_chunks WHERE chunk_id IN ({placeholders})",
                list(chunk_ids))
    rows = cur.fetchall()
    conn.close()
    return [(r["chunk_id"], r["doc_hash"], r["chunk_text"]) for r in rows]
