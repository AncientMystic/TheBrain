#!/usr/bin/env python3
"""
apply_changes.py - Apply project file changes from a manifest.

Usage:
    python apply_changes.py [--manifest updates.json] [--dry-run]

Manifest format (JSON):
{
  "files": [
    {
      "path": "config.py",
      "action": "create",        // create | update | delete
      "content": "..."           // required for create/update
    }
  ]
}
"""

import json
import sys
from pathlib import Path

def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_file(path, content, dry_run=False):
    p = Path(path)
    if dry_run:
        print(f"[DRY-RUN] Write {p}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')
    print(f"✅ Wrote {p}")

def delete_file(path, dry_run=False):
    p = Path(path)
    if dry_run:
        print(f"[DRY-RUN] Delete {p}")
        return
    if p.exists():
        p.unlink()
        print(f"🗑️  Deleted {p}")
    else:
        print(f"⚠️  Missing (skip): {p}")

def apply_manifest(manifest_path, dry_run=False):
    manifest = load_manifest(manifest_path)
    files = manifest.get("files", [])
    for entry in files:
        path = entry["path"]
        action = entry.get("action", "create")
        if action == "delete":
            delete_file(path, dry_run)
        elif action in ("create", "update"):
            content = entry.get("content", "")
            write_file(path, content, dry_run)
        else:
            print(f"❌ Unknown action '{action}' for {path}")

def main():
    manifest_path = "updates.json"
    dry_run = False
    if "--manifest" in sys.argv:
        idx = sys.argv.index("--manifest") + 1
        if idx < len(sys.argv):
            manifest_path = sys.argv[idx]
    if "--dry-run" in sys.argv:
        dry_run = True

    if not Path(manifest_path).exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    apply_manifest(manifest_path, dry_run)

if __name__ == "__main__":
    main()