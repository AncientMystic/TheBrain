from chat.conversation import get_conversation_context
from core import db

def build_context(facts, summaries=None, chunks=None, conversation_history=None, detail_mode=False):
    parts = []
    if conversation_history:
        parts.append(f"[Conversation so far]\n{conversation_history}")
    for fact in facts:
        doc_name = fact.get("doc_name", "unknown")
        source_span = fact.get("source_span", "")
        chunk_id = fact.get("chunk_id", None)
        if chunk_id is not None:
            parts.append(f"[Fact from {doc_name} (chunk {chunk_id})] {fact.get('fact_text')} (source: {source_span})")
        else:
            parts.append(f"[Fact from {doc_name}] {fact.get('fact_text')} (source: {source_span})")
    if summaries:
        for s in summaries:
            parts.append(f"[Summary: {s.get('doc_name','')}] {s.get('summary','')}")
    if chunks:
        # Cache document names to avoid repeated DB lookups
        doc_name_cache = {}
        for _, _, doc_hash, text in chunks:
            if doc_hash not in doc_name_cache:
                conn = db.db_connect("index")
                cur = conn.cursor()
                cur.execute("SELECT filename FROM documents WHERE file_hash=?", (doc_hash,))
                row = cur.fetchone()
                conn.close()
                doc_name_cache[doc_hash] = row["filename"] if row else doc_hash
            doc_name = doc_name_cache[doc_hash]
            parts.append(f"[Chunk from doc {doc_name}] {text[:500]}")
    return "\n\n".join(parts)
