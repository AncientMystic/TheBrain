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

def audit_all():
    print("=== Running Automatic Audit ===")
    actions = cleanup_graph()
    for a in actions:
        print("-", a)
    print("Audit complete.")
