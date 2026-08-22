import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import config
from core import db
from core.recoll_client import run_recoll_query
from extraction.preview_extractor import extract_preview
from ingestion.document_store import store_document, store_chunks
from ingestion.chunker import chunk_document
from core.embeddings import get_embeddings_batch
from extraction.llm_extractor import extract_from_chunks


def store_all_extracted(conn, doc_hash, file_name, all_extracted):
    """Store all categories into key_facts.db."""
    for ent in all_extracted.get("entities", []):
        conn.execute("""
            INSERT INTO entities (doc_hash, entity_type, entity_name, normalized_name, source_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_hash, ent.get("entity_type", "OTHER"), ent.get("entity_name", ""),
              ent.get("normalized_name", ""), ent.get("source_span", ""), ent.get("confidence", 0.0)))
    for person in all_extracted.get("people", []):
        conn.execute("""
            INSERT INTO people (doc_hash, person_name, normalized_name, role, source_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_hash, person.get("person_name", ""), person.get("normalized_name", ""),
              person.get("role", ""), person.get("source_span", ""), person.get("confidence", 0.0)))
    for loc in all_extracted.get("locations", []):
        conn.execute("""
            INSERT INTO locations (doc_hash, location_name, normalized_place, location_type, source_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_hash, loc.get("location_name", ""), loc.get("normalized_place", ""),
              loc.get("location_type", ""), loc.get("source_span", ""), loc.get("confidence", 0.0)))
    for date in all_extracted.get("dates", []):
        conn.execute("""
            INSERT INTO dates (doc_hash, date_text, normalized_date, date_type, source_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_hash, date.get("date_text", ""), date.get("normalized_date", ""),
              date.get("date_type", ""), date.get("source_span", ""), date.get("confidence", 0.0)))
    for event in all_extracted.get("events", []):
        conn.execute("""
            INSERT INTO events (doc_hash, event_name, normalized_name, event_date, event_type,
                                description, significance, source_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_hash, event.get("event_name", ""), event.get("normalized_name", ""),
              event.get("event_date", ""), event.get("event_type", ""), event.get("description", ""),
              event.get("significance", ""), event.get("source_span", ""), event.get("confidence", 0.0)))
    for disc in all_extracted.get("discoveries", []):
        conn.execute("""
            INSERT INTO discoveries (doc_hash, discovery_name, normalized_name, description,
                                     date, significance, source_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (doc_hash, disc.get("discovery_name", ""), disc.get("normalized_name", ""),
              disc.get("description", ""), disc.get("date", ""), disc.get("significance", ""),
              disc.get("source_span", ""), disc.get("confidence", 0.0)))
    for gem in all_extracted.get("gems", []):
        conn.execute("""
            INSERT INTO gems (doc_hash, gem_text, category, importance, source_span, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_hash, gem.get("gem_text", ""), gem.get("category", ""),
              gem.get("importance", 0.0), gem.get("source_span", ""), gem.get("confidence", 0.0)))


def process_recoll_fast(keyword: str, max_results: int = None, preview_chars: int = None):
    """
    Process Recoll results in batches:
    1. Collect all previews.
    2. Store documents/chunks for all previews.
    3. Batch embed all previews and chunks.
    4. Batch run LLM extraction on all chunks.
    5. Store all extracted knowledge.
    """
    if max_results is None:
        max_results = config.RECOLL_MAX_RESULTS
    if preview_chars is None:
        preview_chars = config.PREVIEW_CHAR_WINDOW

    try:
        results = run_recoll_query(keyword, max_results)
    except Exception as e:
        print(f"Error querying Recoll for '{keyword}': {e}")
        return
    print(f"Found {len(results)} Recoll results for '{keyword}'")
    if config.DEBUG_VERBOSE:
        for r in results:
            print('    -', r.get('path'), 'page', r.get('page'))

    # Phase 1: Extract previews and prepare pseudo-documents
    preview_entries = []   # list of (pseudo_hash, filepath, original_name, preview_text, chunks)
    seen_hashes = set()

    for i, res in enumerate(results):
        filepath = res.get("path")
        page = res.get("page")
        if not filepath or not Path(filepath).exists():
            print(f"  [{i+1}/{len(results)}] Skipping missing file: {filepath}")
            continue

        try:
            preview = extract_preview(filepath, keyword, page)
        except Exception as e:
            print(f"Preview extraction error for {filepath}: {e}")
            continue
        if not preview:
            print(f"  [{i+1}/{len(results)}] No preview for {Path(filepath).name}")
            continue

        seed = f"{filepath}:{keyword}:{page or 0}"
        pseudo_hash = "preview:" + hashlib.sha1(seed.encode()).hexdigest()
        if pseudo_hash in seen_hashes:
            continue

        # Dedup from database
        conn_idx = db.db_connect("index")
        cur_idx = conn_idx.cursor()
        cur_idx.execute("SELECT 1 FROM processing_progress WHERE file_hash=?", (pseudo_hash,))
        if cur_idx.fetchone():
            conn_idx.close()
            print(f"  [{i+1}/{len(results)}] Already processed: {Path(filepath).name}")
            continue
        conn_idx.close()

        seen_hashes.add(pseudo_hash)
        chunks = chunk_document(preview)
        if not chunks:
            chunks = [preview]

        original_name = Path(filepath).name
        preview_entries.append((pseudo_hash, filepath, original_name, preview, chunks))
        print(f"  [{i+1}/{len(results)}] Prepared preview for {original_name} ({len(preview)} chars)")

    if not preview_entries:
        print("No new previews to process.")
        return

    # Phase 2: Store documents and chunks in index.db, and create embeddings
    print("Storing documents and chunks...")
    all_previews = [entry[3] for entry in preview_entries]
    all_chunks_flat = []          # list of (doc_hash, chunk_index, chunk_text)
    doc_hashes = [entry[0] for entry in preview_entries]

    # Store documents
    conn_idx = db.db_connect("index")
    for pseudo_hash, filepath, original_name, preview, chunks in preview_entries:
        store_document(conn_idx, pseudo_hash, filepath, f"{original_name} [preview]",
                       "preview", preview, {}, ocr_used=False, page_count=None)
        store_chunks(conn_idx, pseudo_hash, chunks)
        for j, chunk_text in enumerate(chunks):
            all_chunks_flat.append((pseudo_hash, j, chunk_text))
    conn_idx.commit()
    conn_idx.close()

    # Batch embeddings for previews
    print("Generating embeddings for previews...")
    preview_embs = get_embeddings_batch(all_previews, batch_size=config.EMBEDDING_BATCH_SIZE)
    conn_emb = db.db_connect("embeddings")
    for pseudo_hash, emb in zip(doc_hashes, preview_embs):
        if emb:
            blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes())
            conn_emb.execute("INSERT OR REPLACE INTO document_embeddings (doc_hash, embedding, model) VALUES (?,?,?)",
                             (pseudo_hash, blob, config.EMBEDDING_MODEL))
    conn_emb.commit(); conn_emb.close()

    # Batch embeddings for all chunks
    print("Generating embeddings for chunks...")
    all_chunk_texts = [t for _, _, t in all_chunks_flat]
    chunk_embs = get_embeddings_batch(all_chunk_texts, batch_size=config.EMBEDDING_BATCH_SIZE)
    conn_emb = db.db_connect("embeddings")
    cur_emb = conn_emb.cursor()
    conn_idx = db.db_connect("index")
    cur_idx = conn_idx.cursor()
    for (doc_hash, chunk_idx, chunk_text), emb in zip(all_chunks_flat, chunk_embs):
        if emb:
            cur_idx.execute("SELECT chunk_id FROM document_chunks WHERE doc_hash=? AND chunk_index=?",
                            (doc_hash, chunk_idx))
            row = cur_idx.fetchone()
            if row:
                chunk_id = row[0]
                blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes())
                cur_emb.execute("INSERT OR REPLACE INTO chunk_embeddings (chunk_id, doc_hash, chunk_text, embedding, model) VALUES (?,?,?,?,?)",
                                (chunk_id, doc_hash, chunk_text, blob, config.EMBEDDING_MODEL))
    conn_emb.commit(); conn_emb.close(); conn_idx.close()

    # Phase 3: LLM extraction on all chunks in batches
    print(f"Running knowledge extraction on {len(all_chunk_texts)} chunks...")
    old_batch = config.LLM_BATCH_CHUNKS
    config.LLM_BATCH_CHUNKS = config.RECOLL_FAST_LLM_BATCH_CHUNKS if hasattr(config, 'RECOLL_FAST_LLM_BATCH_CHUNKS') else 4
    try:
        chunk_results = extract_from_chunks(all_chunk_texts, model=None, chunk_embeddings=chunk_embs, logic_context="")
    finally:
        config.LLM_BATCH_CHUNKS = old_batch

    # Phase 4: Store extracted knowledge
    print("Storing extracted knowledge...")
    conn_facts = db.db_connect("key_facts")
    cur_facts = conn_facts.cursor()
    conn_idx = db.db_connect("index")
    cur_idx = conn_idx.cursor()

    total_facts = 0
    total_entities = 0

    doc_hash_to_name = {entry[0]: entry[2] for entry in preview_entries}

    for (doc_hash, chunk_idx, chunk_text), chunk_data in zip(all_chunks_flat, chunk_results):
        # Basic validation for facts
        if "facts" in chunk_data:
            cleaned_facts = [f for f in chunk_data["facts"]
                             if f.get("source_span") and f.get("source_span") in chunk_text
                             and f.get("confidence", 0) >= 0.5]
        else:
            cleaned_facts = []

        # Insert facts and sources
        for fact in cleaned_facts:
            cur_facts.execute("""
                INSERT INTO key_facts (doc_hash, doc_name, fact_type, fact_text, canonical_value, source_span, confidence, verified)
                VALUES (?,?,?,?,?,?,?,0)
            """, (doc_hash, f"{doc_hash_to_name.get(doc_hash, 'preview')} [preview]", fact.get("fact_type"), fact.get("fact_text"),
                  fact.get("canonical_value"), fact.get("source_span"), fact.get("confidence", 0.0)))
            fact_id = cur_facts.lastrowid
            # find chunk_id
            cur_idx.execute("SELECT chunk_id FROM document_chunks WHERE doc_hash=? AND chunk_index=?",
                            (doc_hash, chunk_idx))
            row = cur_idx.fetchone()
            chunk_id = row[0] if row else None
            if chunk_id:
                cur_facts.execute("""
                    INSERT INTO fact_sources (fact_id, doc_hash, chunk_id, evidence_span, exact_quote)
                    VALUES (?,?,?,?,?)
                """, (fact_id, doc_hash, chunk_id, fact.get("source_span"), fact.get("source_span")))
            total_facts += 1

        # Insert other categories
        store_all_extracted(cur_facts, doc_hash, f"{doc_hash_to_name.get(doc_hash, 'preview')} [preview]", chunk_data)
        total_entities += len(chunk_data.get("entities", []))

    conn_facts.commit(); conn_facts.close(); conn_idx.close()

    # Mark all processed
    conn_idx = db.db_connect("index")
    cur_idx = conn_idx.cursor()
    now = datetime.now(timezone.utc).isoformat()
    for pseudo_hash, *_ in preview_entries:
        cur_idx.execute("""
            INSERT OR REPLACE INTO processing_progress (file_hash, status, stage, updated_at)
            VALUES (?, 'processed', 'recoll_fast', ?)
        """, (pseudo_hash, now))
    conn_idx.commit(); conn_idx.close()

    print(f"\nRecoll fast processing complete. Processed {len(preview_entries)} previews, extracted {total_facts} facts, {total_entities} entities.")




def collect_seed_keywords(limit=20):
    """Collect distinct keywords from existing key_facts to use as automatic queries."""
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT canonical_value FROM key_facts
        WHERE canonical_value IS NOT NULL AND canonical_value != ''
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    # Also add entities
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT entity_name FROM entities
        WHERE entity_name IS NOT NULL AND entity_name != ''
        LIMIT ?
    """, (limit,))
    rows2 = cur.fetchall()
    conn.close()
    keywords = {row[0] for row in rows if row[0]}
    keywords.update(row[0] for row in rows2 if row[0])
    return sorted(keywords)
