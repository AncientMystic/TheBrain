#!/usr/bin/env python3
"""Export verification standards to JSON facts list."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db

def export_verification_facts(json_path):
    conn = db.db_connect("verification_standards")
    cur = conn.cursor()
    cur.execute("""
        SELECT id, standard_id, statement, subject, predicate, object, negation,
               temporal_bounds_json, truth_status, source_type, source_doc_hash,
               priority, confidence, source_hierarchy_level, data_model_policy,
               psych_score_total, psych_score_breakdown_json, enforcement_vector,
               intentionality_triad_json, lived_experience_cluster,
               funding_gatekeeping_flags_json, socratic_assessment_json,
               supporting_evidence_json, provenance_json, verified_by, verified_at
        FROM verified_standards
        ORDER BY priority ASC, id ASC
    """)
    rows = cur.fetchall()
    conn.close()

    facts = []
    for r in rows:
        facts.append({
            "id": r["standard_id"],
            "statement": r["statement"],
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "negation": r["negation"],
            "temporal_bounds": json.loads(r["temporal_bounds_json"]) if r["temporal_bounds_json"] else {},
            "truth_status": r["truth_status"],
            "source_type": r["source_type"],
            "source_doc_hash": r["source_doc_hash"],
            "priority": r["priority"],
            "confidence": r["confidence"],
            "socratic_metadata": json.loads(r["socratic_assessment_json"]) if r["socratic_assessment_json"] else {},
            "supporting_evidence": json.loads(r["supporting_evidence_json"]) if r["supporting_evidence_json"] else [],
            "provenance": json.loads(r["provenance_json"]) if r["provenance_json"] else {},
            "verified_by": r["verified_by"],
            "verified_at": r["verified_at"],
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "facts": facts}, f, indent=2, ensure_ascii=False)
    print(f"Exported {len(facts)} verification facts to {json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/export_verification_facts.py <json_file>")
        sys.exit(1)
    export_verification_facts(sys.argv[1])
