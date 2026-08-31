
from chat.conversation import get_conversation_context
from core import db
from collections import defaultdict

def build_context(facts, summaries=None, chunks=None, conversation_history=None, detail_mode=False):
    parts = []
    if conversation_history:
        lines = conversation_history.strip().split('\n')
        recent = [l for l in lines if not l.startswith("[Conversation so far]")][-10:]
        if recent:
            parts.append("Recent conversation:\n" + "\n".join(recent))

    facts_by_doc = defaultdict(list)
    for fact in facts:
        doc_name = fact.get("doc_name", "unknown")
        facts_by_doc[doc_name].append(fact)

    doc_numbers = {}
    num = 1
    for doc_name in facts_by_doc:
        doc_numbers[doc_name] = num
        num += 1

    for doc_name, doc_facts in facts_by_doc.items():
        ref_num = doc_numbers[doc_name]
        parts.append(f"### Document [{ref_num}]: {doc_name}")
        for fact in doc_facts:
            source_span = fact.get("source_span", "")
            fact_text = fact.get('fact_text', '')
            confidence = fact.get('confidence', 0.0)
            status = fact.get('verification_status', 'unverified')
            status_str = f"{status}, conf {confidence:.2f}"
            parts.append(f"[{ref_num}] {fact_text} ({status_str}; source: {source_span})")

    if summaries:
        parts.append("\n### Summaries:")
        for s in summaries:
            parts.append(f"- {s.get('doc_name','')}: {s.get('summary','')}")

    if chunks:
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
            if doc_name not in doc_numbers:
                ref_num = len(doc_numbers) + 1
                doc_numbers[doc_name] = ref_num
            ref_num = doc_numbers[doc_name]
            parts.append(f"[{ref_num}] {text[:500]}")

    if doc_numbers:
        parts.append("\n### References")
        for doc_name, ref_num in sorted(doc_numbers.items(), key=lambda x: x[1]):
            parts.append(f"[{ref_num}] {doc_name}")

    return "\n\n".join(parts)
