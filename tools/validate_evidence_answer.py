"""
CI validator for evidence-answer contract fixtures (deterministic, offline).

Validates every fixtures/evidence_answer/*.json against
contracts/evidence_answer.json. Exit nonzero on any violation.
Run: python tools/validate_evidence_answer.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from retrieval.evidence_contract import validate_contract
    fix_dir = ROOT / "fixtures" / "evidence_answer"
    files = sorted(fix_dir.glob("*.json"))
    if not files:
        print("no fixtures found")
        return 1
    failed = 0
    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"FAIL {path.name}: unreadable ({e})")
            failed += 1
            continue
        problems = validate_contract(doc)
        if problems:
            print(f"FAIL {path.name}:")
            for p in problems:
                print(f" - {p}")
            failed += 1
        else:
            print(f"OK {path.name}")
    print(f"{len(files) - failed}/{len(files)} fixtures valid")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
