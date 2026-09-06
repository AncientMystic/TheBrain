"""Static code graphs: deterministic parsing, no fabrications, additive storage."""
from graph.code_graph import parse_code_graph, store_code_graph

SAMPLE = '''
import os
from pathlib import Path


class Builder:
    def build(self, target):
        return compile(target)


def compile(target):
    return os.path.join("out", str(target))


async def fetch(url):
    return compile(url)
'''


def test_parse_shapes():
    g = parse_code_graph(SAMPLE)
    assert ("Builder", 6) in [(n, l) for n, l in g["classes"]]
    assert any(n == "Builder.build" for n, _ in g["functions"])
    assert any(n == "compile" for n, _ in g["functions"])
    assert any(n == "fetch" for n, _ in g["functions"])  # async covered
    assert ("os", 2) in [(m, l) for m, l in g["imports"]]
    assert any(m == "pathlib.Path" for m, _ in g["imports"])
    callers = {c for c, _, _ in g["calls"]}
    assert "Builder.build" in callers and "compile" in callers


def test_unparseable_yields_empty():
    assert parse_code_graph("def broken(:\n  ???") == {"functions": [], "classes": [], "imports": [], "calls": []}
    assert parse_code_graph("") == {"functions": [], "classes": [], "imports": [], "calls": []}


def test_store_and_query():
    from core import db
    doc_hash = "test-code-graph-doc"
    conn = db.db_connect("hypergraph")
    try:
        conn.execute("DELETE FROM edges WHERE doc_hash=?", (doc_hash,))
        conn.execute("DELETE FROM nodes WHERE doc_hash=?", (doc_hash,))
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass
    try:
        n = store_code_graph(doc_hash, parse_code_graph(SAMPLE), "sample.py")
        assert n > 0
        conn = db.db_connect("hypergraph")
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM nodes WHERE doc_hash=? AND node_type='FUNCTION'", (doc_hash,))
            assert cur.fetchone()["n"] >= 3
            cur.execute("""SELECT COUNT(*) AS n FROM edges WHERE doc_hash=? AND relation_type='calls'""", (doc_hash,))
            assert cur.fetchone()["n"] >= 2
        finally:
            try:
                conn.close()
            except Exception:
                pass
    finally:
        conn = db.db_connect("hypergraph")
        try:
            conn.execute("DELETE FROM edges WHERE doc_hash=?", (doc_hash,))
            conn.execute("DELETE FROM nodes WHERE doc_hash=?", (doc_hash,))
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
