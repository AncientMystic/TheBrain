import config
from core import db

def cleanup_graph():
    """
    Flag potentially weak/suspicious keyword-topic edges for review.
    Does NOT delete anything. Prints progress.
    """
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    # Ensure review_flags table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS review_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT,
            entity_id TEXT,
            reason TEXT,
            details TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    actions = []

    # Over-general keywords
    print("  [1/3] Checking over-general keywords...")
    cur.execute("""
        SELECT keyword, COUNT(DISTINCT topic) as topic_count
        FROM keyword_topic_edges
        GROUP BY keyword
        HAVING topic_count > ?
    """, (config.MAX_TOPICS_PER_KEYWORD,))
    over_general = cur.fetchall()
    for i, (keyword, cnt) in enumerate(over_general, 1):
        if i % 10 == 0 or i == len(over_general):
            print(f"    {i}/{len(over_general)} over-general keywords flagged")
        cur.execute("""
            INSERT INTO review_flags (entity_type, entity_id, reason, details)
            VALUES ('keyword_edge', ?, 'over_general', ?)
        """, (keyword, f"Associated with {cnt} topics"))
        actions.append(f"Flagged over-general keyword '{keyword}' ({cnt} topics)")

    # Weak edges
    print("  [2/3] Checking weak edges...")
    cur.execute("""
        SELECT keyword, topic, weight FROM keyword_topic_edges WHERE weight < ?
    """, (config.WEAK_AVG_WEIGHT,))
    weak_edges = cur.fetchall()
    for i, (keyword, topic, weight) in enumerate(weak_edges, 1):
        if i % 20 == 0 or i == len(weak_edges):
            print(f"    {i}/{len(weak_edges)} weak edges flagged")
        cur.execute("""
            INSERT INTO review_flags (entity_type, entity_id, reason, details)
            VALUES ('keyword_edge', ?, 'weak_weight', ?)
        """, (f"{keyword}|{topic}", f"Weight {weight}"))
    actions.append(f"Flagged {len(weak_edges)} weak keyword-topic edges")

    # Short/numeric keywords (preserving years/dates/acronyms)
    print("  [3/3] Checking short/numeric keywords...")
    cur.execute("""
        SELECT keyword, topic, weight FROM keyword_topic_edges
        WHERE (length(keyword) <= 2 AND keyword != upper(keyword))
           OR (keyword GLOB '*[0-9]*' AND NOT (keyword GLOB '[12][0-9][0-9][0-9]' OR keyword GLOB '[12][0-9][0-9][0-9]-[01][0-9]-[0-3][0-9]'))
    """)
    junk_edges = cur.fetchall()
    for i, (keyword, topic, weight) in enumerate(junk_edges, 1):
        if i % 20 == 0 or i == len(junk_edges):
            print(f"    {i}/{len(junk_edges)} short/numeric edges flagged")
        cur.execute("""
            INSERT INTO review_flags (entity_type, entity_id, reason, details)
            VALUES ('keyword_edge', ?, 'short_or_numeric', ?)
        """, (f"{keyword}|{topic}", f"Weight {weight}"))
    actions.append(f"Flagged {len(junk_edges)} short/numeric keyword edges")

    conn.commit()
    conn.close()
    return actions


def audit_against_standards():
    """
    Compare unverified key_facts against verified_standards using dynamic thresholds.
    Prints progress per fact. No deletion.
    """
    from core.embeddings import get_embeddings_batch

    print("  Loading standards...")
    conn_std = db.db_connect("verification_standards")
    cur_std = conn_std.cursor()
    cur_std.execute("SELECT id, statement, subject, predicate, object, negation, truth_status, priority FROM verified_standards")
    standards = [dict(row) for row in cur_std.fetchall()]
    conn_std.close()

    if not standards:
        print("  No verification standards found. Skipping standards comparison.")
        return

    print("  Loading unverified facts...")
    conn_kf = db.db_connect("key_facts")
    cur_kf = conn_kf.cursor()
    cur_kf.execute("SELECT fact_id, fact_text, canonical_value, confidence, verification_status FROM key_facts WHERE verification_status='unverified'")
    facts = [dict(row) for row in cur_kf.fetchall()]
    conn_kf.close()

    if not facts:
        print("  No unverified facts to compare.")
        return

    print(f"  Comparing {len(facts)} facts against {len(standards)} standards...")
    standard_statements = [s["statement"] for s in standards]
    fact_statements = [f["fact_text"] for f in facts]
    standard_embs = get_embeddings_batch(standard_statements, batch_size=config.EMBEDDING_BATCH_SIZE)
    fact_embs = get_embeddings_batch(fact_statements, batch_size=config.EMBEDDING_BATCH_SIZE)

    import numpy as np
    from core.hyperbolic import ensure_hyperbolic, hyperbolic_distance
    all_sims = []
    for fe in fact_embs:
        if fe is None:
            continue
        a = ensure_hyperbolic(fe, space='hyperbolic')
        for se in standard_embs:
            if se is None:
                continue
            b = ensure_hyperbolic(se, space='hyperbolic')
            try:
                d = float(hyperbolic_distance(a, b))
            except Exception:
                continue
            sim = 1.0 / (1.0 + d)
            all_sims.append(sim)
    if not all_sims:
        return

    aligned_threshold = np.percentile(all_sims, 75)
    disputed_threshold = np.percentile(all_sims, 25)

    conn_comp = db.db_connect("verification_standards")
    cur_comp = conn_comp.cursor()
    conn_kf = db.db_connect("key_facts")
    cur_kf = conn_kf.cursor()
    changes = 0

    for idx, (fact, fact_emb) in enumerate(zip(facts, fact_embs), 1):
        if idx % 10 == 0 or idx == len(facts):
            print(f"    Processed {idx}/{len(facts)} facts")
        if fact_emb is None:
            continue
        best_match = None
        best_sim = 0.0
        a = ensure_hyperbolic(fact_emb, space='hyperbolic')
        for std, std_emb in zip(standards, standard_embs):
            if std_emb is None:
                continue
            b = ensure_hyperbolic(std_emb, space='hyperbolic')
            try:
                d = float(hyperbolic_distance(a, b))
            except Exception:
                continue
            sim = 1.0 / (1.0 + d)
            if sim > best_sim:
                best_sim = sim
                best_match = std
        if best_match is not None:
            if best_sim >= aligned_threshold:
                relation = "aligned"
                cur_kf.execute("UPDATE key_facts SET verification_status='aligned', verified_by='standards_audit' WHERE fact_id=?",
                               (fact["fact_id"],))
                changes += 1
            elif best_sim < disputed_threshold:
                relation = "disputed"
                cur_kf.execute("UPDATE key_facts SET verification_status='disputed' WHERE fact_id=?",
                               (fact["fact_id"],))
                changes += 1
            else:
                relation = "unknown"
            if relation in ("aligned", "disputed"):
                cur_comp.execute("""
                    INSERT INTO standard_comparisons (fact_type, fact_id, standard_id, relation, method, confidence)
                    VALUES ('key_fact', ?, ?, ?, 'embedding', ?)
                """, (fact["fact_id"], best_match["id"], relation, best_sim))
    conn_kf.commit(); conn_kf.close()
    conn_comp.commit(); conn_comp.close()
    print(f"  Standards audit compared {len(facts)} facts; updated {changes} statuses.")


def audit_all():
    print("=== Running Automatic Audit ===")
    print("[Step 1/2] Cleaning graph (flag-only)...")
    actions = cleanup_graph()
    for a in actions:
        print("-", a)
    print("[Step 2/2] Comparing against standards...")
    audit_against_standards()
    print("Audit complete.")

