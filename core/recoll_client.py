import re
import subprocess
import shutil
from pathlib import Path
from urllib.parse import urlparse, unquote

import config


class RecollClient:
    """Client wrapper around recollq subprocess."""
    def __init__(self, bin_path=None, db_dir=None, max_results=None):
        self.bin = bin_path or config.RECOLL_BIN
        self.db = db_dir or config.RECOLL_DB
        self.max_results = max_results or config.RECOLL_MAX_RESULTS

    def search(self, query: str, limit: int = None) -> list[dict]:
        return run_recoll_query(query, limit=limit or self.max_results)

    def query(self, query: str, limit: int = None) -> list[dict]:
        return self.search(query, limit)

    def run(self, query: str, limit: int = None) -> list[dict]:
        return self.search(query, limit)


def run_recoll_query(query: str, limit: int = None) -> list[dict]:
    """Run recollq as subprocess and parse output."""
    if limit is None:
        limit = config.RECOLL_MAX_RESULTS

    if not shutil.which(config.RECOLL_BIN):
        raise RuntimeError(f"Recoll binary '{config.RECOLL_BIN}' not found in PATH.")

    cmd = [config.RECOLL_BIN]
    if config.RECOLL_DB:
        cmd += ["-c", config.RECOLL_DB]
    cmd += ["-t", "-n", str(limit), query]

    if config.DEBUG_VERBOSE:
        print(f"    (Running: {' '.join(cmd)})")

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Recoll query timed out after 30 seconds.")

    stdout = proc.stdout.decode('utf-8', errors='replace') if proc.stdout else ''
    if config.DEBUG_VERBOSE:
        print(f"    (recollq stdout raw):\n{stdout[:2000]}")
    stderr = proc.stderr.decode('utf-8', errors='replace') if proc.stderr else ''

    if config.DEBUG_VERBOSE:
        print(f"    (recollq returncode={proc.returncode})")
        print(f"    (recollq stdout preview: {stdout[:500]})")
        print(f"    (recollq stderr preview: {stderr[:500]})")

    if proc.returncode != 0:
        raise RuntimeError(f"Recoll query failed: {stderr.strip()}")

    results = parse_recoll_output(stdout)

    if config.DEBUG_VERBOSE:
        print(f"    (Parsed {len(results)} results)")

    return results


def parse_recoll_output(output: str) -> list[dict]:
    """Parse text output from recollq -t.

    The actual output looks like:
        Recoll query: Query(1996)
        33185 results (printing  5 max):
        text/html\t[file:///H:/path/file.epub]\t[Title]\t201544\tbytes\t
        application/pdf\t[file:///H:/path/file.pdf]\t[Another Title]\t11579711\tbytes\t

    So we must:
      - Skip the first two header lines.
      - Split each result line by tab.
      - Extract the URL from the second field (inside square brackets).
      - Extract the title from the third field (inside square brackets).
    """
    if not output:
        return []

    results = []
    lines = output.splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Skip header lines
        if line.startswith("Recoll query:") or "results (printing" in line:
            continue

        # Expected fields: mime, [url], [title], size, bytes
        fields = line.split("\t")

        # Find the field that contains file:// (usually fields[1])
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
            # Maybe no brackets? Try direct
            url = url_field.strip().strip('[').strip(']').strip()

        # Convert file:// URL to local path
        path_str = url
        if path_str.startswith("file://"):
            # On Windows, file:///C:/... or file://C:/...
            path_str = path_str.replace("file://", "", 1)
            # Remove leading slash for drive letters (file:///C:/ -> C:/)
            if re.match(r'^/[A-Za-z]:/', path_str):
                path_str = path_str[1:]
            path_str = path_str.replace('/', '\\')

        # Extract title from fields[2] if present
        title = ""
        if len(fields) > 2:
            title_raw = fields[2].strip()
            title = title_raw.strip('[]').strip()

        results.append({
            "path": path_str,
            "page": None,          # not present in -t output
            "snippet": "",         # not present in -t output
            "relevancy": 0.0,      # not present in -t output
            "title": title,
        })

    # Normalize paths to absolute
    normalized = []
    for r in results:
        if "path" in r and r["path"]:
            r["path"] = str(Path(r["path"]).resolve())
            normalized.append(r)
    return normalized
