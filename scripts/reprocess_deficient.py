#!/usr/bin/env python3
"""
scripts/reprocess_deficient.py (final corrected)

Reprocess low‑density documents using original extractor.
- No verification, no validation queue, no distilled extractor, no novelty/gate.
- Deletes old data before reprocessing.
- Stores facts, quotes, entities, people, locations, dates, events, discoveries, gems.
- Computes fact embeddings and rebuilds graphs.
- Shows progress per document and chunk.

Usage:
    python scripts/reprocess_deficient.py [--min-facts-per-chunk 0.2] [--limit N] [--dry-run]
"""

import sys
import time
from pathlib import Path
import sqlite3
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import db
from core.progress import ProgressTracker
from core.file_utils import get_file_hash
from core.embeddings import get_embeddings_batch
from extractors.registry import extract_text_from_file
from extractors.pdf_extractor import extract_pdf
from ingestion.chunker import chunk_document
from extraction.llm_extractor import extract_from_chunks
from extraction.summarizer import summarize_document
from graph.hypergraph_builder import build_hypergraph
from graph.external_graph_builder import build_external_graph
import extraction.cleaners as cleaners_mod
import logging
logger = logging.getLogger(__name__)


def find_deficient_documents(min_facts_per_chunk=0.2, limit=None):
    conn_idx = db.db_connect("index")
    cur_idx = conn_idx.cursor()
    cur_idx.execute("SELECT file_hash, filename FROM documents")
    docs = cur_idx.fetchall()
    conn_idx.close()

    conn_kf = db.db_connect("key_facts")
    cur_kf = conn_kf.cursor()

    deficient = []
    for doc in docs:
        doc_hash = doc["file_hash"]
        filename = doc["filename"]
        conn_idx = db.db_connect("index")
        cur_idx = conn_idx.cursor()
        cur_idx.execute("SELECT COUNT(*) FROM document_chunks WHERE doc_hash=?", (doc_hash,))
        chunk_count = cur_idx.fetchone()[0]
        conn_idx.close()
        cur_kf.execute("SELECT COUNT(*) FROM key_facts WHERE doc_hash=?", (doc_hash,))
        fact_count = cur_kf.fetchone()[0]
        if chunk_count > 0 and (fact_count / chunk_count) < min_facts_per_chunk:
            deficient.append((doc_hash, filename, chunk_count, fact_count, fact_count / chunk_count))
    conn_kf.close()
    deficient.sort(key=lambda x: x[4])
    if limit:
        deficient = deficient[:limit]
    return deficient


def table_has_column(conn, table, column):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    return column in cols


def delete_old_data(doc_hash):
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    tables = ["key_facts", "entities", "people", "locations", "dates",
              "events", "discoveries", "gems", "quotes", "fact_sources", "entity_fact_index"]
    for table in tables:
        try:
            if table_has_column(conn, table, "doc_hash"):
                cur.execute(f"DELETE FROM {table} WHERE doc_hash=?", (doc_hash,))
        except Exception as e:
            print(f"    (delete {table} error: {e})")
    conn.commit()
    conn.close()


def reprocess_document(doc_hash, filename, _tracker):
    # Find file path
    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM documents WHERE file_hash=?", (doc_hash,))
    row = cur.fetchone()
    conn.close()
    if not row:
        print(f"  No file_path for {filename}")
        return False
    filepath = Path(row["file_path"])
    if not filepath.exists():
        print(f"  File missing: {filepath}")
        return False

    delete_old_data(doc_hash)

    # Save original settings
    orig_relaxed = cleaners_mod.RELAXED_MODE
    orig_novelty = config.NOVELTY_ENABLED
    orig_gate = config.USE_PRIME_EVEN_GATE
    orig_fast = config.FAST_EXTRACTOR_ENABLED
    orig_validation = config.ENABLE_ASYNC_VALIDATION
    orig_distilled = getattr(config, "USE_DISTILLED_EXTRACTOR", True)
    orig_batch = config.LLM_BATCH_CHUNKS

    # Settings for full extraction using original prompts
    cleaners_mod.RELAXED_MODE = True          # allow more facts
    config.NOVELTY_ENABLED = False
    config.USE_PRIME_EVEN_GATE = False
    config.FAST_EXTRACTOR_ENABLED = False
    config.ENABLE_ASYNC_VALIDATION = False    # no validation queue
    config.USE_DISTILLED_EXTRACTOR = False    # no distilled
    config.LLM_BATCH_CHUNKS = 8               # larger batches

    try:
        # Get text (from cache if available)
        conn = db.db_connect("index")
        cur = conn.cursor()
        cur.execute("SELECT text FROM document_text_cache WHERE file_hash=?", (doc_hash,))
        cache_row = cur.fetchone()
        conn.close()

        if cache_row:
            text = cache_row["text"]
        else:
            suffix = filepath.suffix.lower()
            if suffix == '.pdf':
                result = extract_pdf(filepath)
                text = result["text"]
            else:
                result = extract_text_from_file(filepath)
                text = result["text"]
            conn = db.db_connect("index")
            conn.execute("INSERT OR REPLACE INTO document_text_cache (file_hash, text, metadata_json) VALUES (?,?,?)",
                         (doc_hash, text, "{}"))
            conn.commit()
            conn.close()

        if not text:
            print(f"  No text extracted from {filename}")
            return False

        chunks = chunk_document(text)
        print(f"  Chunks: {len(chunks)}")
        print("  Running original LLM extraction (no verification, no validation, no distilled)...")

        # No chunk_embeddings provided; extract_from_chunks will compute its own if needed
        # but since NOVELTY_ENABLED=False and no gate, it shouldn't need embeddings.
        chunk_results = extract_from_chunks(
            chunks,
            model=None,
            chunk_embeddings=None,
            logic_context="",
        )

        all_extracted = {"facts": [], "entities": [], "relationships": [], "people": [],
                         "locations": [], "dates": [], "events": [], "discoveries": [], "gems": []}
        for ci, cd in enumerate(chunk_results):
            for key in all_extracted:
                if key in cd:
                    all_extracted[key].extend(cd[key])
            if ci % 5 == 0 or ci == len(chunks)-1:
                print(f"    Processed {ci+1}/{len(chunks)} chunks; facts so far: {len(all_extracted['facts'])}")

        # Store facts with NO verification
        conn_facts = db.db_connect("key_facts")
        cur_facts = conn_facts.cursor()
        for fact in all_extracted["facts"]:
            if not fact.get("fact_text"):
                continue
            cur_facts.execute("""
                INSERT INTO key_facts (doc_hash, doc_name, fact_type, fact_text, canonical_value,
                                       source_span, confidence, verification_status, verified_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'unverified', '')
            """, (doc_hash, filename, fact.get("fact_type", "other"), fact.get("fact_text"),
                  fact.get("canonical_value", ""), fact.get("source_span", ""),
                  fact.get("confidence", 0.5)))
            if fact.get("source_span"):
                cur_facts.execute("""
                    INSERT INTO quotes (doc_hash, quote_text, canonical_value, confidence)
                    VALUES (?, ?, ?, ?)
                """, (doc_hash, fact["source_span"], fact.get("canonical_value", ""), fact.get("confidence", 0.0)))
        conn_facts.commit()
        conn_facts.close()

        # Store other categories
        conn_facts = db.db_connect("key_facts")
        for ent in all_extracted["entities"]:
            _insert_entity(conn_facts, doc_hash, filename, ent)
        for person in all_extracted["people"]:
            _insert_person(conn_facts, doc_hash, filename, person)
        for loc in all_extracted["locations"]:
            _insert_location(conn_facts, doc_hash, filename, loc)
        for date in all_extracted["dates"]:
            _insert_date(conn_facts, doc_hash, filename, date)
        for event in all_extracted["events"]:
            _insert_event(conn_facts, doc_hash, filename, event)
        for disc in all_extracted["discoveries"]:
            _insert_discovery(conn_facts, doc_hash, filename, disc)
        for gem in all_extracted["gems"]:
            _insert_gem(conn_facts, doc_hash, filename, gem)
        conn_facts.commit()
        conn_facts.close()

        # Compute fact embeddings
        print("  Computing fact embeddings...")
        conn_facts = db.db_connect("key_facts")
        cur_facts = conn_facts.cursor()
        cur_facts.execute("SELECT fact_id, fact_text FROM key_facts WHERE doc_hash=? AND fact_embedding IS NULL", (doc_hash,))
        facts_rows = cur_facts.fetchall()
        if facts_rows:
            texts = [r["fact_text"] for r in facts_rows]
            embs = get_embeddings_batch(texts, space='hyperbolic')
            for r, emb in zip(facts_rows, embs):
                if emb is not None:
                    blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes())
                    cur_facts.execute("UPDATE key_facts SET fact_embedding=? WHERE fact_id=?", (blob, r["fact_id"]))
        conn_facts.commit()
        conn_facts.close()

        # Rebuild graphs
        print("  Rebuilding hypergraph and external graph...")
        build_hypergraph(doc_hash, all_extracted, {})
        build_external_graph(doc_hash, all_extracted, {})
        print("  Graphs rebuilt.")

        # Summary
        summary, key_points = summarize_document(chunks)
        conn_summ = db.db_connect("summaries")
        conn_summ.execute("INSERT OR REPLACE INTO doc_summaries (doc_hash, doc_name, summary, key_points_json, verification_status) VALUES (?,?,?,?,'unverified')",
                          (doc_hash, filename, summary, str(key_points)))
        conn_summ.commit()
        conn_summ.close()

        print(f"  Complete. Extracted {len(all_extracted['facts'])} facts, {len(all_extracted['entities'])} entities, "
              f"{len(all_extracted['people'])} people, {len(all_extracted['locations'])} locations, "
              f"{len(all_extracted['dates'])} dates, {len(all_extracted['events'])} events, "
              f"{len(all_extracted['discoveries'])} discoveries, {len(all_extracted['gems'])} gems, quotes stored.")
        return True
    finally:
        cleaners_mod.RELAXED_MODE = orig_relaxed
        config.NOVELTY_ENABLED = orig_novelty
        config.USE_PRIME_EVEN_GATE = orig_gate
        config.FAST_EXTRACTOR_ENABLED = orig_fast
        config.ENABLE_ASYNC_VALIDATION = orig_validation
        config.USE_DISTILLED_EXTRACTOR = orig_distilled
        config.LLM_BATCH_CHUNKS = orig_batch


# Helper insert functions
def _insert_entity(conn, doc_hash, _filename, ent):
    conn.execute("""INSERT INTO entities (doc_hash, entity_type, entity_name, normalized_name, source_span, confidence)
                    VALUES (?,?,?,?,?,?)""",
                 (doc_hash, ent.get("entity_type","OTHER"), ent.get("entity_name",""),
                  ent.get("normalized_name",""), ent.get("source_span",""), ent.get("confidence",0.0)))

def _insert_person(conn, doc_hash, _filename, person):
    conn.execute("""INSERT INTO people (doc_hash, person_name, normalized_name, role, source_span, confidence)
                    VALUES (?,?,?,?,?,?)""",
                 (doc_hash, person.get("person_name",""), person.get("normalized_name",""),
                  person.get("role",""), person.get("source_span",""), person.get("confidence",0.0)))

def _insert_location(conn, doc_hash, _filename, loc):
    conn.execute("""INSERT INTO locations (doc_hash, location_name, normalized_place, location_type, source_span, confidence)
                    VALUES (?,?,?,?,?,?)""",
                 (doc_hash, loc.get("location_name",""), loc.get("normalized_place",""),
                  loc.get("location_type",""), loc.get("source_span",""), loc.get("confidence",0.0)))

def _insert_date(conn, doc_hash, _filename, date):
    conn.execute("""INSERT INTO dates (doc_hash, date_text, normalized_date, date_type, source_span, confidence)
                    VALUES (?,?,?,?,?,?)""",
                 (doc_hash, date.get("date_text",""), date.get("normalized_date",""),
                  date.get("date_type",""), date.get("source_span",""), date.get("confidence",0.0)))

def _insert_event(conn, doc_hash, _filename, event):
    conn.execute("""INSERT INTO events (doc_hash, event_name, normalized_name, event_date, event_type,
                                        description, significance, source_span, confidence)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                 (doc_hash, event.get("event_name",""), event.get("normalized_name",""),
                  event.get("event_date",""), event.get("event_type",""), event.get("description",""),
                  event.get("significance",""), event.get("source_span",""), event.get("confidence",0.0)))

def _insert_discovery(conn, doc_hash, _filename, disc):
    conn.execute("""INSERT INTO discoveries (doc_hash, discovery_name, normalized_name, description,
                                             date, significance, source_span, confidence)
                    VALUES (?,?,?,?,?,?,?,?)""",
                 (doc_hash, disc.get("discovery_name",""), disc.get("normalized_name",""),
                  disc.get("description",""), disc.get("date",""), disc.get("significance",""),
                  disc.get("source_span",""), disc.get("confidence",0.0)))

def _insert_gem(conn, doc_hash, _filename, gem):
    conn.execute("""INSERT INTO gems (doc_hash, gem_text, category, importance, source_span, confidence)
                    VALUES (?,?,?,?,?,?)""",
                 (doc_hash, gem.get("gem_text",""), gem.get("category",""),
                  gem.get("importance",0.0), gem.get("source_span",""), gem.get("confidence",0.0)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-facts-per-chunk", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    deficient = find_deficient_documents(min_facts_per_chunk=args.min_facts_per_chunk, limit=args.limit)
    if not deficient:
        print("No deficient documents found.")
        return

    print(f"Found {len(deficient)} documents with low fact density:")
    for doc_hash, filename, chunks, facts, ratio in deficient:
        print(f"  {filename}: {facts} facts / {chunks} chunks = {ratio:.2f}")

    if args.dry_run:
        print("Dry run complete. No documents modified.")
        return

    print("\nStarting reprocessing...")
    tracker = ProgressTracker()
    for idx, (doc_hash, filename, chunks, facts, ratio) in enumerate(deficient, 1):
        print(f"\n[{idx}/{len(deficient)}] Reprocessing: {filename}")
        success = reprocess_document(doc_hash, filename, tracker)
        if not success:
            print("  Failed.")

    print("\nReprocessing complete.")

if __name__ == "__main__":
    main()
