"""Programming + code DB schemas (phase 89).

programming.db: one sphere table per language family (lang_python,
lang_javascript, lang_sql, lang_systems, lang_web) plus lang_universal
and lang_practices. Row: topic, stage (grammar/logic/rhetoric/standard),
rule, example, trivium_ref (canonical back-link), updated_at.
code.db: code_entities (project functions/classes/methods via AST) +
code_edges (caller -> callee) + code_flags (rule violations with
programming.db references). Idempotent. Run: init_code_dbs.py.
"""
import sqlite3
import sys
import time

sys.path.insert(0, "A:/scripts/TheBrain")

LANG_TABLES = ("lang_python", "lang_javascript", "lang_sql", "lang_systems",
               "lang_web", "lang_universal", "lang_practices")

DOCS_SQL = """
CREATE TABLE IF NOT EXISTS {t}(
    topic TEXT PRIMARY KEY,
    stage TEXT NOT NULL DEFAULT 'grammar',
    rule TEXT NOT NULL DEFAULT '',
    example TEXT NOT NULL DEFAULT '',
    trivium_ref TEXT NOT NULL DEFAULT '',
    updated_at INT DEFAULT 0
);
"""

CODE_SQL = """
CREATE TABLE IF NOT EXISTS code_entities(
    path TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL,
    lineno INT DEFAULT 0, signature TEXT DEFAULT '',
    docstring TEXT DEFAULT '', source_hash TEXT DEFAULT '',
    PRIMARY KEY (path, name, kind)
);
CREATE INDEX IF NOT EXISTS idx_ce_kind ON code_entities(kind);
CREATE TABLE IF NOT EXISTS code_edges(
    src_path TEXT NOT NULL, src_name TEXT NOT NULL,
    dst_name TEXT NOT NULL, edge TEXT NOT NULL DEFAULT 'calls',
    PRIMARY KEY (src_path, src_name, dst_name, edge)
);
CREATE TABLE IF NOT EXISTS code_flags(
    path TEXT NOT NULL, name TEXT NOT NULL, rule_topic TEXT NOT NULL,
    detail TEXT DEFAULT '', created_at INT DEFAULT 0,
    PRIMARY KEY (path, name, rule_topic)
);
"""


def init_programming(conn):
    for t in LANG_TABLES:
        conn.execute(DOCS_SQL.format(t=t))
    conn.commit()


def init_code(conn):
    conn.executescript(CODE_SQL)
    conn.commit()


if __name__ == "__main__":
    from core import db as _db
    _pc = _db.db_connect("programming")
    init_programming(_pc)
    _pc.close()
    _cc = _db.db_connect("code")
    init_code(_cc)
    _cc.close()
    print("init_code_dbs: programming spheres + code tables ready")
