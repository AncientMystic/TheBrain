import os
import subprocess
import shutil
import re
from pathlib import Path

import config


class RecollClient:
    """Client wrapper around recollq subprocess."""
    def __init__(self, bin_path=None, db_dir=None, max_results=None):
        self.bin = bin_path or config.RECOLL_BIN
        self.db = db_dir or config.RECOLL_DB
        self.max_results = max_results or config.RECOLL_MAX_RESULTS

    def search(self, query: str, limit: int = None, fetch_text=False) -> tuple[list, int]:
        results = run_recoll_query(query, limit=limit or self.max_results,
                                   bin_path=self.bin, db_dir=self.db)
        return results, len(results)

    def query(self, query: str, limit: int = None, fetch_text=False) -> tuple[list, int]:
        return self.search(query, limit, fetch_text=fetch_text)

    def run(self, query: str, limit: int = None, fetch_text=False) -> tuple[list, int]:
        return self.search(query, limit, fetch_text=fetch_text)

    def close(self):
        """No-op close for API compatibility."""
        pass


def run_recoll_query(query: str, limit: int = None, bin_path: str = None, db_dir: str = None) -> list[dict]:
    """Run recollq as subprocess and parse output (shell=False, validated args)."""
    if limit is None:
        limit = config.RECOLL_MAX_RESULTS
    try:
        limit = int(limit)
    except Exception:
        limit = int(getattr(config, "RECOLL_MAX_RESULTS", 50))
    limit = max(1, min(limit, int(getattr(config, "RECOLL_HARD_MAX_RESULTS", 200))))
    if not isinstance(query, str) or not query.strip():
        return []
    # Generic length cap (not doc-specific) to avoid oversized argv
    max_qlen = int(getattr(config, "RECOLL_MAX_QUERY_CHARS", 500))
    query = query.strip()[:max_qlen]
    bin_path = bin_path or config.RECOLL_BIN
    db_dir = db_dir if db_dir is not None else config.RECOLL_DB
    # Resolve binary without shell: allow absolute path or PATH lookup
    resolved_bin = bin_path if Path(bin_path).is_absolute() else (shutil.which(bin_path) or bin_path)
    if not (shutil.which(resolved_bin) or (Path(resolved_bin).is_file())):
        raise RuntimeError(f"Recoll binary '{bin_path}' not found in PATH.")
    if db_dir:
        # Resolve confdir to absolute to avoid traversal surprises; must exist if given
        db_path = str(Path(db_dir).expanduser())
    else:
        db_path = ""

    cmd = [resolved_bin]
    if db_path:
        cmd += ["-c", db_path]
    cmd += ["-t", "-A", "-n", str(limit), query]

    if config.DEBUG_VERBOSE:
        print(f"    (Running recollq n={limit})")

    timeout = int(getattr(config, "RECOLL_TIMEOUT", 30))
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout, shell=False, text=False)
    stdout = proc.stdout.decode('utf-8', errors='replace') if proc.stdout else ''
    stderr = proc.stderr.decode('utf-8', errors='replace') if proc.stderr else ''

    if config.DEBUG_VERBOSE:
        print(f"    (recollq returncode={proc.returncode})")
        print(f"    (recollq stdout preview: {stdout[:400]})")

    if proc.returncode != 0:
        raise RuntimeError(f"Recoll query failed: {stderr.strip()}")

    return parse_recoll_output(stdout)


def parse_recoll_output(output: str) -> list[dict]:
    """Parse tab-separated output from recollq -t -A."""
    if not output:
        return []

    results = []
    lines = output.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Recoll query:") or "results (printing" in line:
            continue

        fields = line.split("\t")
        url_field = None
        for f in fields:
            f = f.strip()
            if "file://" in f:
                url_field = f
                break

        if not url_field:
            continue

        # Extract URL from inside [ ... ]
        m = re.search(r'\[\s]*(file://[^\]]+)\]', url_field)
        if m:
            url = m.group(1).strip()
        else:
            url = url_field.strip().strip('[').strip(']').strip()

        path_str = url
        if path_str.startswith("file://"):
            path_str = path_str.replace("file://", "", 1)
            if re.match(r'^/[A-Za-z]:/', path_str):
                path_str = path_str[1:]
            # Normalize path separators only on Windows
            if os.name == 'nt':
                path_str = path_str.replace('/', '\\')
            else:
                path_str = path_str.replace('\\', '/')

        title = ""
        snippet = ""
        # Determine fields: with -A, format: mimetype [url] [title] [snippet] size bytes
        if len(fields) > 2:
            title = fields[2].strip().strip('[]').strip()

        # Snippet often fields[3]
        for f in fields[3:]:
            f = f.strip()
            if f and f.lower() != "bytes" and not f.isdigit():
                snippet = f.strip('[]').strip()
                if len(snippet) > 20:
                    break
                else:
                    snippet = ""

        results.append({
            "path": path_str,
            "page": None,
            "snippet": snippet,
            "relevancy": 0.0,
            "title": title,
        })

    normalized = []
    for r in results:
        if "path" in r and r["path"]:
            r["path"] = str(Path(r["path"]).resolve())
            normalized.append(r)
    return normalized
