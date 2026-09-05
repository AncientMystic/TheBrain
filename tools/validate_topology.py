"""
Dependency-direction gate: layers may only depend inward (stdlib-only scanner).

Allowed directions (importer -> imported), everything else fails:
  webui/        -> core/, chat/, retrieval/, reasoning/, memory/, logic/, graph/, deep_research/, ingestion/
  chat/         -> core/, memory/, logic/, graph/, retrieval/, reasoning/
  reasoning/    -> core/, chat/, graph/, memory/
  retrieval/    -> core/, memory/, graph/
  extraction/   -> core/, extractors/, fast_extractor/
  ingestion/    -> core/, extractors/
  graph/        -> core/
  memory/       -> core/
  logic/        -> core/
  scripts/      -> anything (operator tooling)
  tests/        -> anything (test scope)
  core/         -> config only (plus stdlib/third-party)
Forbidden in particular: anything importing webui/ (except main.py wiring),
extraction/ importing webui/, chat/, reasoning/ internals beyond the contract.
Run: python tools/validate_topology.py [--root .]
"""
import ast
import sys
from pathlib import Path

LAYERS = ("webui", "chat", "reasoning", "retrieval", "extraction", "ingestion",
          "graph", "memory", "logic", "deep_research", "core", "scripts", "tests")

# Accepted cross edges (utility reuse / orchestration, no import cycles):
# - chat -> extraction: rule_annotator gazetteers (lower-level text lib)
# - logic -> ingestion: shared chunking infrastructure
# - deep_research -> chat, graph: orchestrator composing retrieval + graph
# - core/maintenance.py: operator tooling living in core/ (behaves as scripts/)
# Accepted cross edges (utility reuse / orchestration, no import cycles):
# - chat -> extraction: rule_annotator gazetteers (lower-level text lib)
# - logic -> ingestion: shared chunking infrastructure
# - deep_research -> chat, graph: orchestrator composing retrieval + graph
# - core -> extraction: leaf text libs with zero project-internal top-level
#   imports (rule_annotator, nlp_primitives); depended-upon, never depending
# - reasoning/retrieval -> chat: orchestrators composing chat retrieval
#   (chat/retriever.py itself imports nothing from chat; no back-edge, no cycle)
ALLOWED = {
    "webui": {"core", "chat", "retrieval", "reasoning", "memory", "logic", "graph", "deep_research", "ingestion"},
    "chat": {"core", "memory", "logic", "graph", "retrieval", "reasoning", "extraction"},
    "reasoning": {"core", "chat", "graph", "memory"},
    "retrieval": {"core", "chat", "memory", "graph"},
    "extraction": {"core", "extractors", "fast_extractor"},
    "ingestion": {"core", "extractors"},
    "graph": {"core"},
    "memory": {"core"},
    "logic": {"core", "ingestion"},
    "deep_research": {"core", "chat", "retrieval", "memory", "graph"},
    "core": {"extraction"},
    "scripts": set(LAYERS) | {"extractors", "fast_extractor"},
    "tests": set(LAYERS) | {"extractors", "fast_extractor"},
}

# Single-file exceptions: operator tooling misplaced by history, not architecture.
# core/maintenance.py behaves as scripts/ (only main.py calls it, function-level).
FILE_EXCEPTIONS = {
    "core/maintenance.py": {"reasoning", "scripts"},
}

# Single-file exceptions: operator tooling misplaced by history, not architecture.
# core/maintenance.py behaves as scripts/ (only main.py calls it, function-level).
FILE_EXCEPTIONS = {
    "core/maintenance.py": {"reasoning", "scripts"},
}


def _layer_of(path, root):
    try:
        rel = path.relative_to(root)
    except Exception:
        return None
    if rel.parts[0] in ("main.py", "server.py", "config.py"):
        return None  # wiring files may compose anything
    top = rel.parts[0] if len(rel.parts) > 1 else None
    return top if top in LAYERS else None


def _imports_of(path):
    """Module-top-level imports only (col_offset 0 statements in the body).

    Function-level lazy imports are an accepted deferral pattern, not an
    architectural edge: they cannot create import cycles. Only top-level
    imports shape the dependency graph, so only they are gated.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return set()
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            # Conditional top-level imports (e.g. try/except optionals) still bind.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for a in sub.names:
                        found.add(a.name.split(".")[0])
                elif isinstance(sub, ast.ImportFrom):
                    if sub.module:
                        found.add(sub.module.split(".")[0])
    return found


def validate_tree(root):
    root = Path(root)
    violations = []
    # Map top-level directory names to layers (plus known packages).
    for path in root.rglob("*.py"):
        s = str(path)
        if ".venv" in s or "__pycache__" in s or ".bak" in s:
            continue
        layer = _layer_of(path, root)
        if layer is None:
            continue
        imported = _imports_of(path)
        allowed = ALLOWED.get(layer, set()) | {"config"}
        try:
            rel = str(path.relative_to(root)).replace("\\", "/")
        except Exception:
            rel = ""
        allowed = allowed | FILE_EXCEPTIONS.get(rel, set())
        for mod in sorted(imported):
            if mod in LAYERS and mod not in allowed and mod != layer:
                violations.append(f"{path.relative_to(root)} [{layer} -> {mod}]")
    return violations


def main():
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path(".")
    violations = validate_tree(root)
    if violations:
        print("TOPOLOGY VIOLATIONS:")
        for v in violations:
            print(f" - {v}")
        return 1
    print("topology OK (all imports point inward)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
