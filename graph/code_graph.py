"""
Static AST code graphs for Python sources (stdlib only, deterministic).

Parses definitions (functions, classes, methods), imports, and call edges into
the existing hypergraph tables — making "what calls X" and "blast radius of Y"
graph traversals instead of keyword luck. Python-only by honest scope (stdlib
ast); other languages keep the text path untouched. Unparseable files yield
empty graphs, never partial fabrications. Confidence is 1.0 throughout:
static structure, not model guesswork.
"""
import ast
import logging

logger = logging.getLogger(__name__)


def _dotted(node):
    """Dotted name for Name/Attribute chains (best-effort, no inference)."""
    try:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        parts.reverse()
        return ".".join(parts) if parts else ""
    except Exception:
        return ""


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []  # (qualified_name, lineno)
        self.classes = []  # (name, lineno)
        self.imports = []  # (dotted_module, lineno)
        self.calls = []  # (caller_qualified, callee_dotted, lineno)
        self._stack = []

    def _current(self):
        return ".".join(self._stack) if self._stack else "<module>"

    def visit_ClassDef(self, node):
        self.classes.append((node.name, node.lineno))
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def _visit_function(self, node):
        self.functions.append((self._current() + "." + node.name if self._stack else node.name,
                               node.lineno))
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append((alias.name.split(".")[0], node.lineno))

    def visit_ImportFrom(self, node):
        base = node.module or ""
        for alias in node.names:
            if alias.name != "*":
                self.imports.append((f"{base}.{alias.name}" if base else alias.name, node.lineno))

    def visit_Call(self, node):
        callee = _dotted(node.func)
        if callee:
            self.calls.append((self._current(), callee, node.lineno))
        self.generic_visit(node)


def parse_code_graph(text):
    """Parse Python source text. Returns dict, empty on syntax errors."""
    empty = {"functions": [], "classes": [], "imports": [], "calls": []}
    if not text:
        return empty
    try:
        tree = ast.parse(text)
    except Exception:
        return empty
    try:
        visitor = _Visitor()
        visitor.visit(tree)
        return {"functions": visitor.functions, "classes": visitor.classes,
                "imports": visitor.imports, "calls": visitor.calls}
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        return empty


def store_code_graph(doc_hash, graph, filename=""):
    """Insert parsed graph into hypergraph tables (additive; existing rows kept).

    Node types FUNCTION/CLASS/MODULE; relations defines/calls/imports with
    confidence 1.0 (deterministic analysis). No-op on empty graphs.
    """
    if not graph or not (graph.get("functions") or graph.get("classes")
                         or graph.get("imports") or graph.get("calls")):
        return 0
    from core import db
    conn = db.db_connect("hypergraph")
    try:
        cur = conn.cursor()
        node_ids = {}

        def _node(node_type, text, span=""):
            key = (node_type, text)
            if key in node_ids:
                return node_ids[key]
            cur.execute("""INSERT INTO nodes
                (doc_hash, node_type, node_text, normalized_name, source_span, confidence)
                VALUES (?,?,?,?,?,?)""",
                (doc_hash, node_type, text[:500], text[:500].lower(), span, 1.0))
            node_ids[key] = cur.lastrowid
            return node_ids[key]

        def _edge(src, tgt, rel, span=""):
            cur.execute("""INSERT OR IGNORE INTO edges
                (doc_hash, source_node_id, target_node_id, relation_type, weight, evidence_span, confidence)
                VALUES (?,?,?,?,?,?,?)""", (doc_hash, src, tgt, rel, 1.0, span, 1.0))

        mod_id = _node("MODULE", filename or doc_hash)
        func_ids = {}
        for name, lineno in graph.get("functions", []):
            fid = _node("FUNCTION", name, f"line {lineno}")
            func_ids.setdefault(name.split(".")[-1], fid)
            _edge(mod_id, fid, "defines", f"line {lineno}")
        for name, lineno in graph.get("classes", []):
            cid = _node("CLASS", name, f"line {lineno}")
            _edge(mod_id, cid, "defines", f"line {lineno}")
        for mod, lineno in graph.get("imports", []):
            mid = _node("MODULE", mod, f"line {lineno}")
            _edge(mod_id, mid, "imports", f"line {lineno}")
        # Resolve callers/callees to defined functions when possible (no duplicate rows);
        # unknown callees (stdlib, third-party) become lightweight reference nodes.
        for caller, callee, lineno in graph.get("calls", []):
            try:
                src = func_ids.get(caller.split(".")[-1], mod_id) if caller != "<module>" else mod_id
                tgt = func_ids.get(callee.split(".")[-1])
                if tgt is None:
                    tgt = _node("FUNCTION", callee, f"line {lineno}")
                _edge(src, tgt, "calls", f"line {lineno}")
            except Exception:
                continue
        conn.commit()
        return len(node_ids)
    finally:
        try:
            conn.close()
        except Exception:
            pass
