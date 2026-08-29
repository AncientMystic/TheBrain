"""
Conversation history management for TheBrain chat.

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

def clear_session(session_id):
    init_conversation_db()
    conn = db.db_connect("memories")
    conn.execute("DELETE FROM conversation_history WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()
