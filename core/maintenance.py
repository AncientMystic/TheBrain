
import time
import threading
import config
from audit.auditor import audit_all
from reasoning.governance import detect_contradictions, resolve_contradictions
from core import db
from pathlib import Path

def build_topic_index_if_needed():
    """Build/update hyperbolic topic index."""
    try:
        from core.streaming_topic_index import build_streaming_topic_index as build_topic_index
        build_topic_index()
    except Exception as e:
        print(f"Topic index build error: {e}")

def optimize_databases():
    """Run PRAGMA optimize on all databases."""
    try:
        for db_type in ["index", "summaries", "key_facts", "embeddings", "hypergraph", "external_graph", "memories", "logic", "reasoning", "verification_standards"]:
            conn = db.db_connect(db_type)
            conn.execute("PRAGMA optimize")
            conn.close()
        print("Databases optimized.")
    except Exception as e:
        print(f"Database optimization error: {e}")

def run_maintenance_once(include_topic_index=True, train_gates=False, train_distilled=False):
    print("Running maintenance...")
    if include_topic_index:
        build_topic_index_if_needed()
    audit_all()
    contradictions = detect_contradictions()
    if contradictions:
        print(f"Found {len(contradictions)} contradictions, moving to review.")
        resolve_contradictions(contradictions)
    try:
        from scripts.consolidate_memories import consolidate
        consolidate()
    except Exception as e:
        print(f"Memory consolidation skipped: {e}")
    optimize_databases()
    if train_gates:
        try:
            from scripts.train_gate import train
            train()
        except Exception as e:
            print(f"Gate training skipped: {e}")
        try:
            from scripts.train_verification_gate import train
            train()
        except Exception as e:
            print(f"Verification gate training skipped: {e}")
    if train_distilled:
        try:
            from scripts.train_distilled_extractor import main as train_distilled
            train_distilled()
        except Exception as e:
            print(f"Distilled training skipped: {e}")
    print("Maintenance complete.")

def maintenance_loop(interval_seconds=3600, **kwargs):
    while True:
        try:
            run_maintenance_once(**kwargs)
        except Exception as e:
            print(f"Maintenance error: {e}")
        time.sleep(interval_seconds)

def start_background_maintenance(interval_seconds=3600, **kwargs):
    if getattr(config, "BACKGROUND_MAINTENANCE", False):
        t = threading.Thread(target=maintenance_loop, args=(interval_seconds,), kwargs=kwargs, daemon=True)
        t.start()
        print(f"Background maintenance started (interval {interval_seconds}s).")
