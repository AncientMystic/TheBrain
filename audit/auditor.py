import config
from core import db

def cleanup_graph():
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    actions = []
    cur.execute("""
        SELECT keyword, COUNT(DISTINCT topic) as topic_count
        FROM keyword_topic_edges
        GROUP BY keyword
        HAVING topic_count > ?
    """, (config.MAX_TOPICS_PER_KEYWORD,))
    rows = cur.fetchall()
    for keyword, cnt in rows:
        cur.execute("DELETE FROM keyword_topic_edges WHERE keyword=?", (keyword,))
        actions.append(f"Removed over-general keyword '{keyword}' ({cnt} topics)")
    cur.execute("DELETE FROM keyword_topic_edges WHERE weight < ?", (config.WEAK_AVG_WEIGHT,))
    deleted = cur.rowcount
    if deleted:
        actions.append(f"Removed {deleted} weak keyword-topic edges")
    cur.execute("DELETE FROM keyword_topic_edges WHERE length(keyword) <= 2 OR keyword GLOB '*[0-9]*'")
    deleted = cur.rowcount
    if deleted:
        actions.append(f"Removed {deleted} short/numeric keyword edges")
    conn.commit(); conn.close()
    return actions

def audit_against_standards():
    """
    Compare unverified key_facts against verified_standards.
    Writes alignments/contradictions to standard_comparisons.
    """
    from core.fact_normalizer import normalize_name
    from core.embeddings import get_embeddings_batch

    conn_std = db.db_connect("verification_standards")
    cur_std = conn_std.cursor()
    cur_std.execute("SELECT id, statement, subject, predicate, object, negation, truth_status, priority FROM verified_standards")
    standards = [dict(row) for row in cur_std.fetchall()]
    conn_std.close()

    if not standards:
        print("  No verification standards found. Skipping standards comparison.")
        return

    conn_kf = db.db_connect("key_facts")
    cur_kf = conn_kf.cursor()
    cur_kf.execute("SELECT fact_id, fact_text, canonical_value, confidence, verification_status FROM key_facts WHERE verification_status='unverified'")
    facts = [dict(row) for row in cur_kf.fetchall()]
    conn_kf.close()

    if not facts:
        print("  No unverified facts to compare.")
        return

    standard_statements = [s["statement"] for s in standards]
    fact_statements = [f["fact_text"] for f in facts]
    standard_embs = get_embeddings_batch(standard_statements, batch_size=config.EMBEDDING_BATCH_SIZE)
    fact_embs = get_embeddings_batch(fact_statements, batch_size=config.EMBEDDING_BATCH_SIZE)

    conn_comp = db.db_connect("verification_standards")
    cur_comp = conn_comp.cursor()
    conn_kf = db.db_connect("key_facts")
    cur_kf = conn_kf.cursor()
    changes = 0

    import numpy as np
    for fact, fact_emb in zip(facts, fact_embs):
        best_match = None
        best_sim = 0.0
        for std, std_emb in zip(standards, standard_embs):
            if fact_emb is None or std_emb is None:
                continue
            a = np.array(fact_emb, dtype=np.float32)
            b = np.array(std_emb, dtype=np.float32)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_match = std
        if best_match is not None and best_sim > 0.92:
            relation = "aligned"
            cur_kf.execute("UPDATE key_facts SET verification_status='aligned', verified_by='standards_audit' WHERE fact_id=?",
                           (fact["fact_id"],))
            changes += 1
        elif best_match is not None and best_sim > 0.60:
            relation = "disputed"
            cur_kf.execute("UPDATE key_facts SET verification_status='disputed' WHERE fact_id=?",
                           (fact["fact_id"],))
            changes += 1
        else:
            continue
        if best_match:
            cur_comp.execute("""
                INSERT INTO standard_comparisons (fact_type, fact_id, standard_id, relation, method, confidence)
                VALUES ('key_fact', ?, ?, ?, 'embedding', ?)
            """, (fact["fact_id"], best_match["id"], relation, best_sim))
    conn_kf.commit(); conn_kf.close()
    conn_comp.commit(); conn_comp.close()
    print(f"  Standards audit compared {len(facts)} facts; updated {changes} statuses.")


def audit_all():
    print("=== Running Automatic Audit ===")
    actions = cleanup_graph()
    for a in actions:
        print("-", a)
    audit_against_standards()
    print("Audit complete.")
