"""
Regenerate golden fixtures after INTENTIONAL changes (never to silence drift).

Usage: python tools/regenerate_golden.py [--write]
Default verifies without writing (CI mode). --write re-pins expected blocks
from current code for INTENTIONAL changes only; review the diff before commit.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX_DIR = ROOT / "fixtures" / "answer_golden"


def build_tagged_case():
    from chat.context_builder import build_tagged_context
    with open(FIX_DIR / "tagged_basic.json", encoding="utf-8") as f:
        fix = json.load(f)
    facts = [dict(x) for x in fix["input"]["facts"]]
    text, ordered, tagmap = build_tagged_context(facts, **fix["input"]["kwargs"])
    return {
        "tags": [x.get("citation_tag") for x in ordered],
        "first_text": ordered[0].get("fact_text") if ordered else "",
        "first_confidence": ordered[0].get("confidence") if ordered else 0,
        "contains": [s for s in fix["expected"]["contains"] if s in text],
        "missing": [s for s in fix["expected"]["contains"] if s not in text],
    }


def main():
    do_write = "--write" in sys.argv
    result = build_tagged_case()
    with open(FIX_DIR / "tagged_basic.json", encoding="utf-8") as f:
        fix = json.load(f)
    ok = (result["tags"] == fix["expected"]["tags"]
          and result["first_text"] == fix["expected"]["first_text"]
          and result["first_confidence"] == fix["expected"]["first_confidence"]
          and not result["missing"])
    print(f"tagged_basic: tags={result['tags']} missing={result['missing']} -> {'MATCH' if ok else 'DRIFT'}")
    if ok:
        print("golden fixtures match (no drift)")
        return 0
    if not do_write:
        print("DRIFT detected: inspect the change, then re-run with --write only if intentional.")
        return 1
    fix["expected"] = {"tags": result["tags"], "first_text": result["first_text"],
                       "first_confidence": result["first_confidence"],
                       "contains": fix["expected"]["contains"]}
    with open(FIX_DIR / "tagged_basic.json", "w", encoding="utf-8") as f:
        json.dump(fix, f, indent=2)
    print("re-pinned tagged_basic.json — review the git diff before commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
