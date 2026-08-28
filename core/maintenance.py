
import time
import threading
import config
from audit.auditor import audit_all
from reasoning.governance import detect_contradictions, resolve_contradictions

def run_maintenance_once():
    print("Running maintenance...")
    audit_all()
    contradictions = detect_contradictions()
    if contradictions:
        print(f"Found {len(contradictions)} contradictions, moving to review.")
        resolve_contradictions(contradictions)
    print("Maintenance complete.")

def maintenance_loop(interval_seconds=3600):
    while True:
        try:
            run_maintenance_once()
        except Exception as e:
            print(f"Maintenance error: {e}")
        time.sleep(interval_seconds)

def start_background_maintenance(interval_seconds=3600):
    if getattr(config, "BACKGROUND_MAINTENANCE", False):
        t = threading.Thread(target=maintenance_loop, args=(interval_seconds,), daemon=True)
        t.start()
        print(f"Background maintenance started (interval {interval_seconds}s).")
