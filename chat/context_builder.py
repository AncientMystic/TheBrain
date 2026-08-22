def build_context(facts, summaries=None, chunks=None):
    parts = []
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
        for _, _, doc_hash, text in chunks:
            parts.append(f"[Chunk from doc {doc_hash}] {text[:500]}")
    return "\n\n".join(parts)
