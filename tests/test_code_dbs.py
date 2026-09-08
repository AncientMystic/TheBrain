"""Code DB proof (phase 89): ingest + rule analysis on synthetic files."""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, "A:/scripts/TheBrain")
sys.path.insert(0, "A:/scripts/TheBrain/scripts")

DIRTY = '''
"""Module docstring."""
import os

def clean_function(x):
    """Does the thing properly."""
    return x + 1

def bare_handler(data):
    try:
        return int(data)
    except:
        pass

def runner(cmd):
    return eval(cmd)

API_TOKEN = "sk-live-12345"

def query_all():
    return db.execute("SELECT * FROM users")

# TODO refactor this later
def undocumented(a, b):
    return a * b
'''

CLEAN = '''
"""Module docstring."""

def add(a, b):
    """Add two numbers."""
    return a + b
'''


def test_code_dbs():
    from ingest_code import ingest, analyze
    d = os.path.join(tempfile.gettempdir(), "opencode", "codedb_test")
    os.makedirs(d, exist_ok=True)
    dirty = os.path.join(d, "dirty_mod.py")
    clean = os.path.join(d, "clean_mod.py")
    open(dirty, "w").write(DIRTY)
    open(clean, "w").write(CLEAN)
    import init_code_dbs  # noqa: proves init module imports cleanly
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE code_entities(path TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL,
    lineno INT DEFAULT 0, signature TEXT DEFAULT '', docstring TEXT DEFAULT '',
    source_hash TEXT DEFAULT '', PRIMARY KEY (path, name, kind))""")
    conn.execute("""CREATE TABLE code_edges(src_path TEXT NOT NULL, src_name TEXT NOT NULL,
    dst_name TEXT NOT NULL, edge TEXT NOT NULL DEFAULT 'calls',
    PRIMARY KEY (src_path, src_name, dst_name, edge))""")
    conn.execute("""CREATE TABLE code_flags(path TEXT NOT NULL, name TEXT NOT NULL, rule_topic TEXT NOT NULL,
    detail TEXT DEFAULT '', created_at INT DEFAULT 0,
    PRIMARY KEY (path, name, rule_topic))""")
    try:
        e, x = ingest([dirty, clean], conn=conn)
        assert e >= 5, e
        # analyze needs programming conn only for rule text (report); flags write to code conn
        import types
        n = analyze([dirty, clean], conn=conn, pconn=conn)
        topics = {r[0] for r in conn.execute("SELECT DISTINCT rule_topic FROM code_flags")}
        for t in ("missing-docstring", "bare-except", "eval-exec",
                  "hardcoded-secret", "select-star", "todo-no-ref"):
            assert t in topics, (t, topics)
        dirty_flags = {r[0] for r in conn.execute(
            "SELECT DISTINCT rule_topic FROM code_flags WHERE path=?", (dirty,))}
        assert len(dirty_flags) == 6, dirty_flags
        clean_flags = conn.execute(
            "SELECT COUNT(*) FROM code_flags WHERE path=?", (clean,)).fetchone()[0]
        assert clean_flags == 0, clean_flags
    finally:
        conn.close()
    print("test_code_dbs: OK (6 violation classes fire, clean file silent)")


if __name__ == "__main__":
    test_code_dbs()
