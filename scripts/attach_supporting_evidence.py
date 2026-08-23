#!/usr/bin/env python3
"""Attach aligned facts/data to verified standards as supporting evidence."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db
from core.fact_normalizer import normalize_name


def attach_supporting_evidence_to_standards():
    conn_std = db.db_connect("verification_standards")
    cur_std = conn_std.cursor()
    cur_std.execute("""
        SELECT id, statement, subject, predicate, object
        FROM verified_standards
    """)
    standards = [dict(row) for row in cur_std.fetchall()]
    conn_std.close()

    if not standards:
        print("No standards found. Nothing to attach.")
        return

    conn_kf = db.db_connect("key_facts")
    cur_kf = conn_kf.cursor()
    conn_std = db.db_connect("verification_standards")
    cur_std = conn_std.cursor()

    updated = 0
    for std in standards:
        # Find facts whose normalized statement or canonical value matches
        statement_norm = normalize_name(std["statement"])
        cur_kf.execute("""
            SELECT fact_id, doc_hash, fact_text, canonical_value, confidence
            FROM key_facts
            WHERE verification_status IN ('verified_true', 'admin_claim', 'aligned')
              AND (
                    LOWER(fact_text) = LOWER(?)
                    OR LOWER(canonical_value) = LOWER(?)
                    OR LOWER(canonical_value) = LOWER(?)
                  )
            ORDER BY confidence DESC
            LIMIT 25
        """, (
            std["statement"],
            std.get("subject", ""),
            std.get("object", ""),
        ))
        rows = cur_kf.fetchall()
        evidence = [dict(row) for row in rows]

        if evidence:
            cur_std.execute(
                "UPDATE verified_standards SET supporting_evidence_json=? WHERE id=?",
                (json.dumps(evidence, default=str), std["id"]),
            )
            updated += 1

    conn_kf.close()
    conn_std.commit()
    conn_std.close()
    print(f"Attached supporting evidence to {updated}/{len(standards)} standards.")


if __name__ == "__main__":
    attach_supporting_evidence_to_standards()
