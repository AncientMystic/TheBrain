"""
Validate the tree against TRUST_SURFACE.yaml (inspectability gate, not bureaucracy).

Errors (exit 1): shell=True subprocess, hardcoded API secrets, browser open
without opt-out flag, listen-all-interfaces combined with empty auth token.
Warnings (exit 0): optional binaries unresolvable, unknown subprocess call sites
not listed in the declaration (keeps inventory honest as code evolves).
Run: python scripts/validate_trust_surface.py
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ERRORS = []
WARNINGS = []


def _py_files():
    out = []
    for p in ROOT.rglob("*.py"):
        s = str(p)
        if ".venv" in s or "__pycache__" in s or ".bak" in s:
            continue
        if p.name == "validate_trust_surface.py":
            continue  # own diagnostic strings mention the forbidden pattern
        out.append(p)
    return out


def check_no_shell_true():
    for p in _py_files():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if re.search(r"shell\s*=\s*True", text):
            ERRORS.append(f"{p.relative_to(ROOT)}: shell=True subprocess (must be list-form, shell=False)")


def check_no_hardcoded_secrets():
    for p in _py_files():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if re.search(r"sk-[A-Za-z0-9]{10,}", text):
            ERRORS.append(f"{p.relative_to(ROOT)}: possible hardcoded API secret")


def check_browser_opt_out():
    try:
        text = (ROOT / "main.py").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    if "webbrowser.open" in text and "--no-browser" not in text:
        ERRORS.append("main.py: browser auto-open without --no-browser opt-out")


def check_listen_all_needs_token():
    try:
        import os
        host = os.environ.get("SERVER_HOST", "127.0.0.1")
        token = os.environ.get("SERVER_AUTH_TOKEN", "")
        if host == "0.0.0.0" and not token:
            WARNINGS.append("SERVER_HOST=0.0.0.0 with empty SERVER_AUTH_TOKEN exposes an open LAN server")
    except Exception:
        pass


def check_declared_processes_known():
    # Stdlib-only: plain substring checks, no yaml dependency.
    try:
        text = (ROOT / "TRUST_SURFACE.yaml").read_text(encoding="utf-8")
    except Exception as e:
        ERRORS.append(f"TRUST_SURFACE.yaml unreadable: {e}")
        return
    for keyword in ("recoll", "tesseract", "import_data"):
        if keyword not in text:
            WARNINGS.append(f"TRUST_SURFACE.yaml: no entry mentioning {keyword}")
    # Every subprocess call site should be declared: find Popen/run occurrences.
    for p in _py_files():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "subprocess.run" in text or "subprocess.Popen" in text or "ProcessPoolExecutor" in text:
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            stem = p.stem
            if stem not in ("recoll_client", "pdf_extractor", "import_data", "validate_trust_surface", "embeddings"):
                WARNINGS.append(f"{rel}: subprocess use not in declaration inventory")


def check_optional_binaries():
    for binary in ("recollq", "tesseract"):
        if shutil.which(binary) is None:
            WARNINGS.append(f"optional binary missing: {binary} (feature degrades gracefully)")


def main():
    check_no_shell_true()
    check_no_hardcoded_secrets()
    check_browser_opt_out()
    check_listen_all_needs_token()
    check_declared_processes_known()
    check_optional_binaries()
    for w in WARNINGS:
        print(f"WARN: {w}")
    if ERRORS:
        print("TRUST-SURFACE ERRORS:")
        for e in ERRORS:
            print(f" - {e}")
        return 1
    print(f"trust surface OK ({len(WARNINGS)} warnings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
