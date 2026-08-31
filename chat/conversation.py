"""
Conversation history management for TheBrain chat.

from extraction.summarizer import summarize_chunk
Stores messages per session, provides recent context,
and optionally summarizes older turns to fit token limits.
"""
import sqlite3
import time
from datetime import datetime, timezone
from core import db
import config


def init_conversation_db():
    conn = db.db_connect("memories")  # reuse memories.db for now
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id, timestamp)")
    conn.commit()
    conn.close()

def add_message(session_id, role, content):
    """Add a message to conversation history."""
    init_conversation_db()
    conn = db.db_connect("memories")
    conn.execute(
        "INSERT INTO conversation_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, time.time())
    )
    conn.commit()
    conn.close()

def get_recent_messages(session_id, max_turns=10):
    """Return last max_turns messages (user and assistant) as list of dict."""
    init_conversation_db()
    conn = db.db_connect("memories")
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content FROM (
            SELECT role, content, timestamp
            FROM conversation_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ) ORDER BY timestamp ASC
    """, (session_id, max_turns * 2))  # each turn has user + assistant
    rows = cur.fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def get_conversation_context(session_id, max_turns=None):
    """Build a context string from recent conversation messages."""

    if getattr(config, "USE_HYPERBOLIC_CONVERSATION_SUMMARY", True):
        from chat.conversation_summarizer import get_hyperbolic_conversation_context
        return get_hyperbolic_conversation_context(session_id, max_recent=5, max_clusters=3)

    """Build a context string from recent conversation, optionally summarizing older."""
    if max_turns is None:
        max_turns = config.CONVERSATION_MAX_TURNS
    messages = get_recent_messages(session_id, max_turns)
    parts = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}")
    return "\n\n".join(parts)


def get_hyperbolic_conversation_history(session_id, query, max_turns=10, max_tokens=300,
                                        full_memory_triggers=("recall", "remember", "full history", "everything")):
    """Return only conversation turns hyperbolically relevant to the current query.
       Long messages are summarized while preserving code blocks and key details.
       If query contains full_memory_triggers, returns full relevant messages instead of summarised.
    """
    import re
    import numpy as np
    from core.hyperbolic import hyperbolic_distance
    from core.embeddings import get_embedding
    from core.dynamic_hyperbolic import dynamic_radius

    messages = get_recent_messages(session_id, max_turns)
    if not messages:
        return ""

    q_emb = get_embedding(query, space='hyperbolic')
    if q_emb is None:
        return ""

    request_full = any(trigger in query.lower() for trigger in full_memory_triggers)

    scored = []
    for msg in messages:
        text = msg["content"]
        if len(text) == 0:
            continue
        emb = get_embedding(text[:1000], space='hyperbolic')
        if emb is None:
            continue
        dist = hyperbolic_distance(q_emb, emb)
        scored.append((dist, msg["role"], text))

    if not scored:
        return ""

    distances = [d for d, _, _ in scored]
    k = min(3, len(distances))
    if k == 0:
        return ""
    kth_dist = sorted(distances)[k-1]
    radius = 1.2 * kth_dist if kth_dist > 0 else 1.0

    relevant = []
    for dist, role, text in scored:
        if dist <= radius:
            role_label = "User" if role == "user" else "Assistant"
            if request_full:
                relevant.append(f"{role_label}: {text}")
            else:
                processed_text = _process_message_for_context(text, max_tokens=max_tokens)
                relevant.append(f"{role_label}: {processed_text}")

    if relevant:
        return "\n\n".join(relevant)
    return ""


def _process_message_for_context(text, max_tokens=300):
    """Truncate or summarize message while preserving code blocks and key details."""
    import re
    # Extract code blocks (```...```)
    code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
    # Remove code blocks from text for summarization
    text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    if len(text) <= max_tokens * 4:
        return text
    if len(text_without_code) > max_tokens * 4:
        try:
            summary = summarize_chunk(text_without_code)
            if summary:
                text_without_code = summary
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Conversation truncation error: {e})")
            text_without_code = text_without_code[:max_tokens*4] + "..."
    else:
        text_without_code = text_without_code.strip()

    code_str = "\n\n".join(code_blocks)
    if code_str:
        return f"{text_without_code}\n\n[Code blocks preserved]:\n{code_str}"
    return text_without_code

def clear_session(session_id):
    init_conversation_db()
    conn = db.db_connect("memories")
    conn.execute("DELETE FROM conversation_history WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()
