#!/usr/bin/env python3
"""
Import facts and/or logic training data into TheBrain.

Usage:
    python import_data.py --facts facts.json
    python import_data.py --logic logic.json
    python import_data.py --facts facts.json --logic logic.json

The script uses the existing TheBrain importers:
- Facts are imported via scripts/import_verification_facts.py
- Logic modules are converted to temporary Markdown and imported via main.py --logic --input <temp_dir>
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def safe_name(text: str, max_len: int = 60) -> str:
    """Return a filesystem-safe name from arbitrary text."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", text or "item").strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] or "item"


def import_facts(facts_path: Path, dry_run: bool = False) -> bool:
    """Import facts JSON via existing verification facts importer."""
    if not facts_path.exists():
        print(f"[ERROR] Facts file not found: {facts_path}")
        return False

    if dry_run:
        with open(facts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        facts = data.get("facts", [])
        print(f"[DRY-RUN] Would import {len(facts)} facts from {facts_path}")
        for i, fact in enumerate(facts[:10], 1):
            print(f"  {i}. {fact.get('statement', '')[:120]}")
        if len(facts) > 10:
            print(f"  ... and {len(facts)-10} more")
        return True

    print(f"Importing facts from: {facts_path}")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "import_verification_facts.py"), str(facts_path)],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print("[ERROR] Facts import failed.")
        return False
    print("Facts import completed.")
    return True


def build_logic_markdown(data: dict | list) -> list[tuple[str, str]]:
    """Convert logic JSON to list of (filename, markdown_content)."""
    docs = []

    if isinstance(data, list):
        modules = data
    else:
        modules = data.get("logic_modules", [])
        examples_only = data.get("logic_examples", [])

        if examples_only:
            grouped = {}
            for ex in examples_only:
                cat = safe_name(ex.get("category", "general"), 40)
                grouped.setdefault(cat, []).append(ex)
            for cat, exs in grouped.items():
                parts = [f"# Logic Examples - {cat}\n"]
                for i, ex in enumerate(exs, 1):
                    parts.append(f"## Example {i}\n")
                    parts.append(f"Input:\n{ex.get('input_text','')}\n")
                    parts.append(f"Output:\n{ex.get('output_text','')}\n")
                docs.append((f"examples_{cat}.md", "\n".join(parts)))
            return docs

    for idx, mod in enumerate(modules, 1):
        name = mod.get("name") or f"Logic Module {idx}"
        category = mod.get("category", "reasoning")
        summary = mod.get("summary", "")
        content = mod.get("content", "")
        keywords = mod.get("keywords", [])
        examples = mod.get("examples", [])

        parts = [f"# {name}\n"]
        if category:
            parts.append(f"Category: {category}\n")
        if summary:
            parts.append(f"\n{summary}\n")
        if content:
            parts.append(f"\n{content}\n")

        if examples:
            parts.append("\n## Examples\n")
            for i, ex in enumerate(examples, 1):
                parts.append(f"\n### Example {i}\n")
                parts.append(f"Input:\n{ex.get('input_text','')}\n")
                parts.append(f"Output:\n{ex.get('output_text','')}\n")

        filename = f"{idx:03d}_{safe_name(name)}.md"
        docs.append((filename, "\n".join(parts)))

    return docs


def import_logic(logic_path: Path, dry_run: bool = False, keep_temp: bool = False) -> bool:
    """Import logic training data from JSON."""
    if not logic_path.exists():
        print(f"[ERROR] Logic file not found: {logic_path}")
        return False

    with open(logic_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = build_logic_markdown(data)

    if dry_run:
        print(f"[DRY-RUN] Would create {len(docs)} logic document(s) from {logic_path}")
        for fname, content in docs:
            print(f"  - {fname} ({len(content)} chars)")
        return True

    if not docs:
        print("[ERROR] No valid logic modules or examples found.")
        return False

    print(f"Importing logic from: {logic_path}")
    print(f"Prepared {len(docs)} logic document(s).")

    with tempfile.TemporaryDirectory(prefix="thebrain_logic_") as tmpdir:
        tmp = Path(tmpdir)
        for fname, content in docs:
            path = tmp / fname
            path.write_text(content, encoding="utf-8")
            print(f"  Wrote {fname}")

        print(f"Running logic learning on temporary directory: {tmp}")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "main.py"), "--logic", "--input", str(tmp)],
            cwd=str(REPO_ROOT),
        )

        if keep_temp:
            print(f"[KEEP-TEMP] Temporary directory left at: {tmp}")

    if result.returncode != 0:
        print("[ERROR] Logic import failed.")
        return False
    print("Logic import completed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Import facts or logic training data into TheBrain.")
    parser.add_argument("--facts", type=str, default=None, help="Path to facts JSON file")
    parser.add_argument("--logic", type=str, default=None, help="Path to logic JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show what would be imported without writing")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary logic files after import")
    args = parser.parse_args()

    if not args.facts and not args.logic:
        parser.print_help()
        return 1

    success = True

    if args.facts:
        success = import_facts(Path(args.facts), dry_run=args.dry_run) and success

    if args.logic:
        success = import_logic(Path(args.logic), dry_run=args.dry_run, keep_temp=args.keep_temp) and success

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())