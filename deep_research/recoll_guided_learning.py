"""
Autonomous guided learning using Recoll to fill knowledge gaps.
"""
import time
import json
import os
from pathlib import Path
from datetime import datetime
import config
from core import db
from core.llm import call_model, call_model_json
from core.recoll_client import RecollClient
from deep_research import gap_analyzer


from urllib.parse import urlparse, unquote
import re

def _url_to_path(url):
    """Convert a file:// URL to a local filesystem path."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.scheme == 'file':
            path = unquote(parsed.path)
            # On Windows, remove leading slash before drive letter
            if re.match(r'^/[A-Za-z]:', path):
                path = path[1:]
            return path
        else:
            # Not a file URL; maybe already a path
            return url
    except Exception:
        return None

QUERY_GENERATION_PROMPT = """
We need to find documents that can fill this knowledge gap:
- Type: {gap_type}
- Entity/Topic: {entity}
- Current info: {details}

Suggest a Recoll search query that would retrieve documents containing more information about this entity or its relationships.
Return only the query string (no quotes, no extra text).
"""

def _generate_query_for_gap(gap):
    """Use LLM to create a Recoll query for a given gap."""
    details = ""
    if gap["type"] == "low_confidence":
        details = f"Fact text: {gap.get('text','')} (confidence {gap.get('confidence',0)})"
    elif gap["type"] == "sparse_entity":
        details = f"Currently has {gap.get('edge_count',0)} relationships"
    elif gap["type"] == "low_coverage_keyword":
        details = f"Only {gap.get('fact_count',0)} supporting facts"
    elif gap["type"] == "unconnected_topic":
        details = "No cross-document links"
    else:
        details = "Unknown gap type"

    prompt = QUERY_GENERATION_PROMPT.format(
        gap_type=gap["type"],
        entity=gap.get("entity",""),
        details=details
    )
    query = call_model(prompt, max_tokens=100).strip()
    return query

def _log_query(query_text, purpose):
    conn = db.db_connect("recoll_log")
    cur = conn.cursor()
    cur.execute("INSERT INTO recoll_queries (query_text, purpose) VALUES (?, ?)", (query_text, purpose))
    query_id = cur.lastrowid
    conn.commit()
    conn.close()
    return query_id

def _log_result(query_id, doc_hash, file_path):
    conn = db.db_connect("recoll_log")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO recoll_query_results (query_id, doc_hash, file_path)
        VALUES (?, ?, ?)
    """, (query_id, doc_hash, file_path))
    conn.commit()
    conn.close()

def _mark_processed(query_id, doc_hash, success):
    conn = db.db_connect("recoll_log")
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO recoll_log (query_id, doc_hash, processed_successfully, processed_at)
        VALUES (?, ?, ?, ?)
    """, (query_id, doc_hash, 1 if success else 0, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def _is_duplicate_query(query_text):
    """Return True only if this query has previously produced at least one successfully processed document."""
    conn = db.db_connect("recoll_log")
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*)
        FROM recoll_queries q
        JOIN recoll_log l ON q.id = l.query_id
        WHERE q.query_text = ? AND l.processed_successfully = 1
    """, (query_text,))
    count = cur.fetchone()[0]
    conn.close()
    return count > 0

def run_recoll_guided_learning(process_file_callback, tracker, max_rounds=None, interactive=None):
    """Main autonomous loop."""
    if max_rounds is None:
        max_rounds = config.RECOLL_MAX_ROUNDS
    if interactive is None:
        interactive = config.RECOLL_INTERACTIVE

    # Initialize Recoll log database
    from scripts.init_recoll_log_db import init_recoll_log_db
    init_recoll_log_db()

    # Connect to Recoll
    try:
        recoll_client = RecollClient()
    except ImportError as e:
        print(f"Recoll not available: {e}")
        return

    print(f"Starting Recoll-guided autonomous learning (max {max_rounds} rounds).")

    for round_num in range(1, max_rounds + 1):
        print(f"\n=== Round {round_num} ===")
        gaps = gap_analyzer.get_all_gaps(limit_per_type=5)
        if not gaps:
            print("No knowledge gaps found. Exiting.")
            break

        # Prioritize gaps: missing_relationships and under_explored_topic first
        priority_order = {
            "missing_relationships": 0,
            "under_explored_topic": 1,
            "low_confidence": 2,
            "sparse_entity": 3,
            "low_coverage_keyword": 4,
            "unconnected_topic": 5,
        }
        gaps.sort(key=lambda g: priority_order.get(g.get("type"), 9))

        new_docs_processed = 0
        for gap in gaps:
            entity = gap.get("entity", "unknown")
            print(f"  Gap: {gap['type']} on '{entity}'")
            query = _generate_query_for_gap(gap)
            if not query or _is_duplicate_query(query):
                print("    Skipping duplicate or empty query.")
                continue

            purpose = f"{gap['type']}:{entity}"
            query_id = _log_query(query, purpose)
            print(f"    Query: {query}")

            results, count = recoll_client.search(query, limit=5, fetch_text=False)
            print(f"    Recoll returned {count} results, processing up to {len(results)}")
            for doc in results:
                file_url = doc.get("path") or doc.get("url", "")
                file_path = _url_to_path(file_url)
                if not file_path or not Path(file_path).exists():
                    if config.DEBUG_VERBOSE:
                        print(f"      Skipping missing/non-file URL: {file_url}")
                    continue
                # Check if already processed by TheBrain
                from core.file_utils import get_file_hash
                file_hash = get_file_hash(file_path)
                if tracker.is_processed(file_hash):
                    print(f"      Already processed: {file_path}")
                    continue
                # Check recoll_log for duplicate
                conn = db.db_connect("recoll_log")
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM recoll_log WHERE query_id=? AND doc_hash=?", (query_id, file_hash))
                if cur.fetchone()[0] > 0:
                    conn.close()
                    print(f"      Already queried: {file_path}")
                    continue
                conn.close()

                if interactive:
                    answer = input(f"      Process document '{file_path}'? (y/n): ").strip().lower()
                    if answer != 'y':
                        continue

                print(f"      Processing: {file_path}")
                success = process_file_callback(Path(file_path), tracker)
                _mark_processed(query_id, file_hash, success)
                if success:
                    new_docs_processed += 1
                time.sleep(0.5)  # small delay

        print(f"  Round {round_num} processed {new_docs_processed} new documents.")
        if new_docs_processed == 0:
            print("No new documents were processed this round. Stopping to avoid loops.")
            break

    recoll_client.close()
    print("Recoll-guided learning complete.")
