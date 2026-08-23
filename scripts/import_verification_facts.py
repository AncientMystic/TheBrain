#!/usr/bin/env python3
"""Import verification facts from JSON into verification_standards.db."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import db
from core.fact_normalizer import normalize_name

def import_verification_facts(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    facts = data.get("facts", [])
    conn = db.db_connect("verification_standards")
    cur = conn.cursor()
    for f in facts:
        statement = f.get("statement", "")
        subject = f.get("subject", "")
        predicate = f.get("predicate", "")
        obj = f.get("object", "")
        negation = int(f.get("negation", 0) or 0)
        standard_id = f.get("id") or f"json-{normalize_name(statement)[:20]}"
        cur.execute("""
            INSERT OR REPLACE INTO verified_standards
            (standard_id, statement, subject, predicate, object, negation,
             truth_status, source_type, priority, confidence, socratic_assessment_json,
             supporting_evidence_json, provenance_json, verified_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'admin_claim', 0, ?, ?, ?, ?, 'json_import')
        """, (
            standard_id, statement, subject, predicate, obj, negation,
            f.get("truth_status", "admin_claim"),
            f.get("confidence", 1.0),
            json.dumps(f.get("socratic_metadata", {})),
            json.dumps(f.get("supporting_evidence", [])),
            json.dumps(f.get("provenance", {})),
        ))
    conn.commit()
    conn.close()
    print(f"Imported {len(facts)} verification facts.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/import_verification_facts.py <json_file>")
        sys.exit(1)
    import_verification_facts(sys.argv[1])
