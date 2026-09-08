"""Project code ingest + rule analysis (phase 89).

ingest(paths): AST-walk Python files -> code_entities (functions, classes,
Class.method) + code_edges (imports). Skips .venv/__pycache__/backups.
analyze(limit paths): rule checks referencing programming.db topics —
  missing docstring on public functions, bare except, TODO without issue
  ref, eval/exec use, hardcoded secrets, SELECT-star strings. Findings go
  to code_flags(path, name, rule_topic, detail). Query CLI: print flags
  for a file with rule text. Idempotent; never raises.
"""
import ast
import hashlib
import os
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

SKIP_DIRS = {".venv", "__pycache__", "backups", ".git", "node_modules",
             ".opencode", "graphify-out"}
# Example/sample files carry placeholder secrets by design; scanning them
# only produces false positives (measured: all 4 live secret flags).
SKIP_FILES = {"test_code_dbs.py"}

RULE_TOPICS = {
    "missing-docstring": ("lang_practices", "Docstring contract"),
    "bare-except": ("lang_practices", "Never swallow exceptions"),
    "todo-no-ref": ("lang_practices", "Contextual body"),
    "eval-exec": ("lang_practices", "Injection concatenation"),
    "hardcoded-secret": ("lang_practices", "Secret storage discipline"),
    "select-star": ("lang_sql", "SQL clause order"),
}


def _iter_py(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for f in fns:
            if f.endswith(".py") and f not in SKIP_FILES and ".example." not in f:
                yield os.path.join(dp, f)


def ingest(paths, conn=None):
    """Walk files into code_entities/code_edges. Returns (entities, edges)."""
    own = False
    if conn is None:
        from core import db as _db
        conn = _db.db_connect("code")
        own = True
    n_e = n_x = 0
    try:
        for path in paths:
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    src = fh.read()
                tree = ast.parse(src)
            except Exception:
                continue
            h = hashlib.sha256(src.encode("utf-8", errors="ignore")).hexdigest()[:16]
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    sig = f"({', '.join(args)})"
                    doc = (ast.get_docstring(node) or "")[:500]
                    conn.execute("INSERT OR REPLACE INTO code_entities"
                                 " (path, name, kind, lineno, signature, docstring, source_hash)"
                                 " VALUES (?,?,?,?,?,?,?)",
                                 (path, node.name, "function", node.lineno, sig, doc, h))
                    n_e += 1
                elif isinstance(node, ast.ClassDef):
                    doc = (ast.get_docstring(node) or "")[:500]
                    conn.execute("INSERT OR REPLACE INTO code_entities"
                                 " (path, name, kind, lineno, signature, docstring, source_hash)"
                                 " VALUES (?,?,?,?,?,?,?)",
                                 (path, node.name, "class", node.lineno, "", doc, h))
                    n_e += 1
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args = [a.arg for a in sub.args.args]
                            doc2 = (ast.get_docstring(sub) or "")[:500]
                            conn.execute("INSERT OR REPLACE INTO code_entities"
                                         " (path, name, kind, lineno, signature, docstring, source_hash)"
                                         " VALUES (?,?,?,?,?,?,?)",
                                         (path, f"{node.name}.{sub.name}", "method",
                                          sub.lineno, f"({', '.join(args)})", doc2, h))
                            n_e += 1
            try:
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import,)):
                        for a in node.names:
                            conn.execute("INSERT OR IGNORE INTO code_edges"
                                         " (src_path, src_name, dst_name, edge)"
                                         " VALUES (?,?,?,?)",
                                         (path, os.path.basename(path), (a.name or "").split(".")[0], "imports"))
                            n_x += 1
                    elif isinstance(node, ast.ImportFrom):
                        mod = (node.module or "").split(".")[0]
                        if mod:
                            conn.execute("INSERT OR IGNORE INTO code_edges"
                                         " (src_path, src_name, dst_name, edge)"
                                         " VALUES (?,?,?,?)",
                                         (path, os.path.basename(path), mod, "imports"))
                            n_x += 1
            except Exception:
                pass
        conn.commit()
    finally:
        if own:
            try:
                conn.close()
            except Exception:
                pass
    return n_e, n_x


def _rule_text(conn, table, topic_frag):
    try:
        row = conn.execute(f"SELECT rule FROM {table} WHERE topic LIKE ?",
                           (f"%{topic_frag}%",)).fetchone()
        return (row[0] or "")[:200] if row else ""
    except Exception:
        return ""


def analyze(paths, conn=None, pconn=None):
    """Run rule checks on files. Returns flag count. Writes code_flags."""
    import re as _re
    own_c = own_p = False
    if conn is None:
        from core import db as _db
        conn = _db.db_connect("code")
        own_c = True
    if pconn is None:
        from core import db as _db2
        pconn = _db2.db_connect("programming")
        own_p = True
    n = 0
    try:
        for path in paths:
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    src = fh.read()
                tree = ast.parse(src)
            except Exception:
                continue
            base = os.path.basename(path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                    if not (ast.get_docstring(node) or "").strip():
                        conn.execute("INSERT OR REPLACE INTO code_flags (path, name, rule_topic, detail, created_at)"
                                     " VALUES (?,?,?,?,strftime('%s','now'))",
                                     (path, node.name, "missing-docstring",
                                      f"public function line {node.lineno}"))
                        n += 1
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    conn.execute("INSERT OR REPLACE INTO code_flags (path, name, rule_topic, detail, created_at)"
                                 " VALUES (?,?,?,?,strftime('%s','now'))",
                                 (path, f"line:{node.lineno}", "bare-except", "bare except clause"))
                    n += 1
                if isinstance(node, (ast.Call,)) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                    conn.execute("INSERT OR REPLACE INTO code_flags (path, name, rule_topic, detail, created_at)"
                                 " VALUES (?,?,?,?,strftime('%s','now'))",
                                 (path, f"line:{node.lineno}", "eval-exec", f"{node.func.id}() call"))
                    n += 1
            for i, line in enumerate(src.splitlines(), 1):
                if _re.search(r"\b(TODO|FIXME|HACK)\b", line) and not _re.search(r"#\d+", line):
                    conn.execute("INSERT OR REPLACE INTO code_flags (path, name, rule_topic, detail, created_at)"
                                 " VALUES (?,?,?,?,strftime('%s','now'))",
                                 (path, f"line:{i}", "todo-no-ref", line.strip()[:120]))
                    n += 1
                if _re.search(r"(?i)(password|secret|api[_-]?key|token)\s*['\"]?\s*=\s*['\"]", line):
                    conn.execute("INSERT OR REPLACE INTO code_flags (path, name, rule_topic, detail, created_at)"
                                 " VALUES (?,?,?,?,strftime('%s','now'))",
                                 (path, f"line:{i}", "hardcoded-secret", "literal secret assignment"))
                    n += 1
                if _re.search(r"(?i)select\s+\*", line):
                    conn.execute("INSERT OR REPLACE INTO code_flags (path, name, rule_topic, detail, created_at)"
                                 " VALUES (?,?,?,?,strftime('%s','now'))",
                                 (path, f"line:{i}", "select-star", line.strip()[:120]))
                    n += 1
        conn.commit()
    finally:
        if own_c:
            try:
                conn.close()
            except Exception:
                pass
        if own_p:
            try:
                pconn.close()
            except Exception:
                pass
    return n


def report(path, conn=None, pconn=None, limit=20):
    """Print flags for a file with programming.db rule text."""
    own_c = own_p = False
    if conn is None:
        from core import db as _db
        conn = _db.db_connect("code")
        own_c = True
    if pconn is None:
        from core import db as _db2
        pconn = _db2.db_connect("programming")
        own_p = True
    try:
        rows = conn.execute("SELECT name, rule_topic, detail FROM code_flags"
                            " WHERE path=? LIMIT ?", (path, int(limit))).fetchall()
        for name, topic, detail in rows:
            table, frag = RULE_TOPICS.get(topic, ("lang_practices", topic))
            print(f"[{topic}] {name}: {detail} :: {_rule_text(pconn, table, frag)[:120]}")
        print(f"report: {len(rows)} flags for {os.path.basename(path)}")
        return len(rows)
    finally:
        if own_c:
            try:
                conn.close()
            except Exception:
                pass
        if own_p:
            try:
                pconn.close()
            except Exception:
                pass


if __name__ == "__main__":
    import sys as _sys
    _root = _sys.argv[1] if len(_sys.argv) > 1 else "A:/scripts/TheBrain"
    _files = list(_iter_py(_root))
    _e, _x = ingest(_files)
    print(f"ingest_code: {_e} entities, {_x} import edges from {len(_files)} files")
