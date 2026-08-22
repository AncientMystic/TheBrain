import config
from core import db
from core.llm import call_model


def llm_audit_keywords(limit=50):
    """
    Ask an LLM to review suspicious keywords and remove those it deems garbage.
    """
    conn = db.db_connect("external_graph")
    cur = conn.cursor()
    cur.execute("""
        SELECT keyword, COUNT(DISTINCT topic) as topic_count
        FROM keyword_topic_edges
        WHERE length(keyword) <= 2 OR keyword GLOB '*[0-9]*'
        GROUP BY keyword
        ORDER BY topic_count DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    if not rows:
        conn.close()
        return []

    keywords = [row[0] for row in rows]
    prompt = (
        "You are a knowledge graph auditor.\n"
        "Review the following keywords that are suspicious because they are very short, numeric, or associated with many topics.\n"
        "Identify which keywords are likely garbage or should be removed from the graph.\n"
        "Output only the keywords to remove, one per line.\n"
        "Do not include any explanations or extra text.\n\n"
        "Keywords:\n" + "\n".join(keywords)
    )

    response = call_model(prompt, model=config.MODEL_NAME, max_tokens=256)
    to_remove = [line.strip() for line in response.splitlines() if line.strip()]
    if to_remove:
        for kw in to_remove:
            cur.execute("DELETE FROM keyword_topic_edges WHERE keyword=?", (kw,))
        conn.commit()
        conn.close()
        return to_remove
    conn.close()
    return []