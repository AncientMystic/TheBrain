#!/usr/bin/env python3
"""
scripts/audit_extraction_coverage.py

Comprehensive audit of extracted knowledge coverage per document.
Checks counts for all knowledge categories and flags low density or missing data.

Usage:
    python scripts/audit_extraction_coverage.py [--min-items-per-chunk 0.5] [--show-samples]
"""

import sys
import time
from pathlib import Path
import sqlite3
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import db
from core.text_utils import tokenize

# Categories and their table/column mapping
CATEGORIES = {
    "facts": ("key_facts", "key_facts", "doc_hash"),
    "entities": ("key_facts", "entities", "doc_hash"),
    "people": ("key_facts", "people", "doc_hash"),
    "locations": ("key_facts", "locations", "doc_hash"),
    "dates": ("key_facts", "dates", "doc_hash"),
    "events": ("key_facts", "events", "doc_hash"),
    "discoveries": ("key_facts", "discoveries", "doc_hash"),
    "gems": ("key_facts", "gems", "doc_hash"),
    "quotes": ("key_facts", "quotes", "doc_hash"),
}

def _get_chunks_for_doc(doc_hash):
    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, chunk_text FROM document_chunks WHERE doc_hash=?", (doc_hash,))
    rows = cur.fetchall()
    conn.close()
    return rows

def get_category_counts(doc_hash):
    counts = {}
    for cat, (db_name, table, col) in CATEGORIES.items():
        conn = db.db_connect(db_name)
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col}=?", (doc_hash,))
        counts[cat] = cur.fetchone()[0]
        conn.close()
    return counts

def get_chunk_count(doc_hash):
    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_chunks WHERE doc_hash=?", (doc_hash,))
    cnt = cur.fetchone()[0]
    conn.close()
    return cnt

def get_text_length(doc_hash):
    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("SELECT text_length FROM documents WHERE file_hash=?", (doc_hash,))
    row = cur.fetchone()
    conn.close()
    return row["text_length"] if row else 0

def check_span_validity(doc_hash, chunks):
    """Check fact source_spans against chunks."""
    conn = db.db_connect("key_facts")
    cur = conn.cursor()
    cur.execute("SELECT fact_id, source_span FROM key_facts WHERE doc_hash=?", (doc_hash,))
    rows = cur.fetchall()
    conn.close()
    invalid = []
    for r in rows:
        if not r["source_span"]:
            invalid.append((r["fact_id"], "empty"))
        else:
            span = r["source_span"].lower()
            found = any(span in c["chunk_text"].lower() for c in chunks)
            if not found:
                invalid.append((r["fact_id"], "not_found"))
    return invalid

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-items-per-chunk", type=float, default=0.5,
                        help="Minimum total extracted items per chunk to consider adequate (default 0.5)")
    parser.add_argument("--show-samples", action="store_true",
                        help="Show examples of malformed entries")
    args = parser.parse_args()

    print("=== Extraction Coverage Audit ===")
    t0 = time.time()

    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("SELECT file_hash, filename FROM documents ORDER BY filename")
    docs = cur.fetchall()
    conn.close()

    total = len(docs)
    print(f"Total documents: {total}")

    issues = {}
    for doc in tqdm(docs, desc="Auditing", unit="doc"):
        doc_hash = doc["file_hash"]
        doc_name = doc["filename"]
        chunks = _get_chunks_for_doc(doc_hash)
        chunk_count = len(chunks)
        if chunk_count == 0:
            issues[doc_name] = ["No chunks found"]
            continue

        counts = get_category_counts(doc_hash)
        total_items = sum(counts.values())

        # Density check
        density = total_items / chunk_count
        doc_issues = []
        if density < args.min_items_per_chunk:
            doc_issues.append(f"Low extraction density: {total_items} items / {chunk_count} chunks = {density:.2f}")

        # Per-category zero check (only flag if document likely has content)
        text_len = get_text_length(doc_hash)
        if text_len > 500:  # ignore very short docs
            for cat, cnt in counts.items():
                if cnt == 0:
                    doc_issues.append(f"No {cat} extracted")

        # Span validity for facts
        if counts.get("facts", 0) > 0:
            invalid_spans = check_span_validity(doc_hash, chunks)
            if invalid_spans:
                doc_issues.append(f"{len(invalid_spans)} facts with invalid source spans")
                if args.show_samples and len(invalid_spans) <= 5:
                    for fid, reason in invalid_spans:
                        doc_issues.append(f"    Fact {fid}: {reason}")

        # Malformed entries check (empty text fields)
        for cat, (db_name, table, col) in CATEGORIES.items():
            if cat == "quotes":
                text_col = "quote_text"
            elif cat == "facts":
                text_col = "fact_text"
            elif cat == "entities":
                text_col = "entity_name"
            elif cat == "people":
                text_col = "person_name"
            elif cat == "locations":
                text_col = "location_name"
            elif cat == "dates":
                text_col = "date_text"
            elif cat == "events":
                text_col = "event_name"
            elif cat == "discoveries":
                text_col = "discovery_name"
            elif cat == "gems":
                text_col = "gem_text"
            else:
                continue
            conn = db.db_connect(db_name)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {col}=? AND ({text_col} IS NULL OR {text_col}='')", (doc_hash,))
            empty = cur.fetchone()[0]
            conn.close()
            if empty > 0:
                doc_issues.append(f"{empty} {cat} with empty {text_col}")

        if doc_issues:
            issues[doc_name] = doc_issues

    if issues:
        print(f"\n=== Issues found in {len(issues)} documents ===")
        for name, iss in issues.items():
            print(f"\nDocument: {name}")
            for i in iss:
                print(f"  - {i}")
    else:
        print("\nNo issues found. Extraction coverage looks adequate.")

    print(f"\nAudit completed in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()