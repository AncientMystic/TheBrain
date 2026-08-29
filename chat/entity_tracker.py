
import json
from core import db
from chat.query_analyzer import analyze_query
from core.text_utils import tokenize

def extract_active_entities(answer_text, top_k=10):
    """Extract important entities from assistant answer as list of strings."""
    if not answer_text:
        return []
    analysis = analyze_query(answer_text)
    entities = set()
    # Collect from annotations
    for ent in analysis.get("entities", []):
        text = ent.get("text") if isinstance(ent, dict) else str(ent)
        if text and len(text) > 2:
            entities.add(text)
    # Also include long tokens (potential concepts)
    tokens = tokenize(answer_text)
    for t in tokens:
        if len(t) > 4:
            entities.add(t)
    # Sort by length desc and return top_k
    sorted_entities = sorted(entities, key=len, reverse=True)[:top_k]
    return sorted_entities

def save_active_entities(session_id, entities):
    """Store active entities JSON in memory_sessions."""
    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO memory_sessions (session_id, active_entities_json)
        VALUES (?, ?)
        ON CONFLICT(session_id) DO UPDATE SET active_entities_json = excluded.active_entities_json
    """, (session_id, json.dumps(entities)))
    conn.commit()
    conn.close()

def load_active_entities(session_id):
    """Load active entities list from memory_sessions."""
    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("SELECT active_entities_json FROM memory_sessions WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    conn.close()
    if row and row["active_entities_json"]:
        try:
            return json.loads(row["active_entities_json"])
        except Exception:
            return []
    return []

def is_anaphoric(query, active_entities, session_centroid=None, query_embedding=None):
    """Return True if query is a continuation referencing previous context."""
    if not active_entities:
        return False
    query_lower = query.lower()
    # Pronoun-based detection
    pronouns = ["these", "those", "it", "they", "them", "this", "that", "the above", "the following", "the latter"]
    if any(p in query_lower for p in pronouns):
        return True
    # Hyperbolic distance to session centroid
    if session_centroid is not None and query_embedding is not None:
        from core.hyperbolic import hyperbolic_distance
        d = hyperbolic_distance(query_embedding, session_centroid)
        if d < 1.0:  # threshold; may need tuning
            return True
    return False
