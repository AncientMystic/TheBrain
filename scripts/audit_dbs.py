"""DB audit (phase 89): schema-vs-code alignment for the 7 working DBs.

Parses CREATE TABLE statements from scripts/init_schemas.py (the code's
declared schema) and diffs against live PRAGMA table_info per database:
missing tables, missing columns, plus row counts and empty-table flags.
--fix runs the matching init_* functions (all IF NOT EXISTS) and ALTERs
in missing columns with declared types. Data rows are never deleted;
anomalies print for human judgment. Read-only by default.

Usage: audit_dbs.py [--fix] [--db alias]
Covers: summaries, key_facts, hypergraph, external_graph, logic,
memories, reasoning.
"""
import re
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

INIT_FUNCS = {
    "summaries": "init_summaries_db",
    "key_facts": "init_key_facts_db",
    "hypergraph": "init_hypergraph_db",
    "external_graph": "init_external_graph_db",
    "logic": "init_logic_db",
    "memories": "init_memories_db",
    "reasoning": "init_reasoning_db",
}


def _split_top_commas(body):
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def parse_expected():
    """{alias: {table: [(col, type)]}} from init_schemas.py source."""
    import scripts.init_schemas as _IS  # noqa: ensure importable
    src = open(_IS.__file__, encoding="utf-8", errors="ignore").read()
    out = {}
    # split source by init function to attribute tables per DB
    chunks = re.split(r"\ndef (init_\w+_db)\(\):", src)
    # chunks[0] preamble, then alternating name/body
    for i in range(1, len(chunks) - 1, 2):
        fname, body = chunks[i], chunks[i + 1]
        alias = None
        for a, f in INIT_FUNCS.items():
            if f == fname:
                alias = a
        if alias is None:
            continue
        tables = {}
        for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+)\s*\(",
                             body):
            tname = m.group(1)
            # find matching close paren from m.end()-1
            start = m.end() - 1
            depth = 0
            end = start
            for j in range(start, len(body)):
                if body[j] == "(":
                    depth += 1
                elif body[j] == ")":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            cols = []
            for part in _split_top_commas(body[start + 1:end]):
                toks = part.strip().split()
                if not toks:
                    continue
                if toks[0].upper() in ("PRIMARY", "FOREIGN", "UNIQUE", "CHECK",
                                       "CONSTRAINT"):
                    continue
                cols.append((toks[0].strip('"`[]'), " ".join(toks[1:])))
            tables[tname] = cols
        out[alias] = tables
    return out


def audit(alias=None, fix=False):
    from core import db as _db
    expected = parse_expected()
    aliases = [alias] if alias else list(INIT_FUNCS)
    summary = {"missing_tables": [], "missing_columns": [], "empty": [],
               "counts": {}}
    for a in aliases:
        exp = expected.get(a, {})
        try:
            conn = _db.db_connect(a)
        except Exception as e:
            print(f"[{a}] CONNECT FAILED: {e}")
            continue
        try:
            live = {r[0] for r in
                    conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for t, cols in exp.items():
                if t not in live:
                    summary["missing_tables"].append(f"{a}.{t}")
                    print(f"[{a}] MISSING TABLE {t} ({len(cols)} cols)")
                    continue
                have = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
                for cname, ctype in cols:
                    if cname not in have:
                        summary["missing_columns"].append(f"{a}.{t}.{cname}")
                        print(f"[{a}] MISSING COLUMN {t}.{cname} {ctype[:40]}")
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    summary["counts"][f"{a}.{t}"] = n
                    if n == 0:
                        summary["empty"].append(f"{a}.{t}")
                except Exception:
                    pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
    print(f"audit: {len(summary['missing_tables'])} missing tables,"
          f" {len(summary['missing_columns'])} missing columns,"
          f" {len(summary['empty'])} empty tables")
    if fix and (summary["missing_tables"] or summary["missing_columns"]):
        _apply_fix(summary)
    return summary


def _apply_fix(summary):
    import scripts.init_schemas as _IS
    ran = set()
    for item in summary["missing_tables"]:
        a = item.split(".")[0]
        if a not in ran:
            try:
                getattr(_IS, INIT_FUNCS[a])()
                print(f"[fix] ran {INIT_FUNCS[a]}")
            except Exception as e:
                print(f"[fix] {INIT_FUNCS[a]} FAILED: {e}")
            ran.add(a)
    if summary["missing_columns"]:
        from core import db as _db
        expected = parse_expected()
        for item in summary["missing_columns"]:
            a, t, cname = item.split(".")
            try:
                ctype = dict(expected.get(a, {}).get(t, [])).get(cname, "TEXT")
                conn = _db.db_connect(a)
                try:
                    conn.execute(f"ALTER TABLE {t} ADD COLUMN {cname} {ctype}")
                    conn.commit()
                    print(f"[fix] {item} added")
                finally:
                    conn.close()
            except Exception as e:
                print(f"[fix] {item} FAILED: {e}")
    # re-run inits for column-only drift (IF NOT EXISTS = safe no-op)
    for a in list(INIT_FUNCS):
        if a not in ran:
            try:
                getattr(_IS, INIT_FUNCS[a])()
            except Exception:
                pass
    print("[fix] done; re-run audit to confirm")


if __name__ == "__main__":
    _fix = "--fix" in sys.argv
    _dbonly = None
    for _a in sys.argv[1:]:
        if _a.startswith("--db"):
            _dbonly = _a.split("=", 1)[1] if "=" in _a else None
    audit(alias=_dbonly, fix=_fix)
