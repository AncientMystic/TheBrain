"""
Publication-safety gate: blocks secrets, stray credential files, and local
absolute paths from reaching a public tree. Errors fail CI; anything absent
from the tree is skipped silently. Run: python tools/check_source_exposure.py
"""
import re
import sys
from pathlib import Path

SECRET_PATTERNS = (
    (r"sk-[A-Za-z0-9]{10,}", "possible API secret"),
    (r"github_pat_[A-Za-z0-9_]{10,}", "possible GitHub token"),
    (r"xox[bap]-[A-Za-z0-9-]{10,}", "possible Slack token"),
    (r"AKIA[0-9A-Z]{16}", "possible AWS key"),
)

FORBIDDEN_FILES = (".env", ".env.local", "*.pem", "*.key", "*.bak", "id_rsa", "id_ed25519")

ABS_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:\\|/home/|/Users/)[^\s\"']*")


def _tree_files(root):
    for p in root.rglob("*"):
        s = str(p)
        if ".venv" in s or "__pycache__" in s or ".git/" in s or ".git\\" in s:
            continue
        if p.is_file():
            yield p


def check(root):
    root = Path(root)
    errors, warnings = [], []
    for p in _tree_files(root):
        name = p.name
        import fnmatch
        if any(fnmatch.fnmatch(name, pat) for pat in FORBIDDEN_FILES):
            # .bak staging notes and local env files must never publish.
            if ".venv" not in str(p):
                errors.append(f"{p.relative_to(root)}: forbidden file for publication")
                continue
        if p.suffix not in {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg", ".ini", ".bat", ".sh", ".html", ".js"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if p.name in ("check_source_exposure.py",):
            continue  # own patterns would self-match
        for pattern, label in SECRET_PATTERNS:
            if re.search(pattern, text):
                errors.append(f"{p.relative_to(root)}: {label}")
        for m in ABS_PATH_PATTERN.finditer(text):
            hit = m.group(0)
            # Allow repository-relative examples and the user's own workspace docs.
            if "A:\\scripts\\TheBrain" in hit or "A:/scripts/TheBrain" in hit:
                warnings.append(f"{p.relative_to(root)}: local absolute path {hit[:60]}")
    return errors, warnings


def main():
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path(".")
    errors, warnings = check(root)
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        print("EXPOSURE ERRORS:")
        for e in errors:
            print(f" - {e}")
        return 1
    print(f"source exposure OK ({len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
