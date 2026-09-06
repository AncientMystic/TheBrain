import json
from reasoning.decompose import decompose_query
from reasoning.agents import MindMapAgent, KGQueryAgent, SourceCheckerAgent, ContradictionDetectorAgent, LogicVerifierAgent
from reasoning.governance import compute_confidence, store_provenance
from core.query_analyzer import analyze_query
from chat.retriever import retrieve_from_graph, fallback_to_chunks
from chat.context_builder import build_context
from core.llm import call_model, call_model_json
from graph.graph_queries import get_related_keywords, get_facts_by_keyword, get_global_node_edges
from graph.expansion import pre_select_candidates
from core import db
import config
import logging
logger = logging.getLogger(__name__)


SUFFICIENCY_PROMPT = """
Given the user question and the information currently available, determine if there is enough information to answer the question completely and accurately.

If YES, return exactly:
{"sufficient": true, "missing": ""}

If NO, return:
{"sufficient": false, "missing": "brief description of what additional information is needed"}
"""

def is_sufficient(query, context):
    prompt = SUFFICIENCY_PROMPT + f"\n\nQuestion: {query}\n\nAvailable information:\n{context}\n"
    data = call_model_json(prompt, max_tokens=128)
    if data and data.get("sufficient"):
        return True
    return False


def extract_keywords_from_fact(fact):
    """Extract potential keywords from fact text or canonical value using tokenizer."""
    from core.text_utils import tokenize, get_bigrams
    text = fact.get("fact_text", "")
    val = fact.get("canonical_value", "")
    combined = text + " " + val
    tokens = tokenize(combined)
    return tokens[:5] + list(get_bigrams(tokens))[:3]


def _analyze_many(texts, memo):
    """analyze_query per unique text, memoized across expansion rounds.

    Rounds re-scan all_facts[:20], so without memo the same fact pays ONNX
    NER once per round (up to 3x per query). Misses run in a small pool on
    CPU-provider sessions only (DML sessions crash on concurrent Run);
    otherwise serial. Identical outputs either way.
    """
    missing = [t for t in dict.fromkeys(texts) if t not in memo]
    if missing:
        _workers = 1
        try:
            import config as _cfg
            _w = max(1, int(getattr(_cfg, "FAST_EXTRACTOR_WORKERS", 4)))
            from core.query_analyzer import _get_fast_extractor as _gfe
            _fx = _gfe()
            _sess = getattr(getattr(_fx, "onnx_extractor", None), "session", None)
            _provs = [str(p) for p in (_sess.get_providers() or [])] if _sess is not None else []
            if _w > 1 and _provs == ["CPUExecutionProvider"]:
                _workers = min(_w, len(missing))
        except Exception:
            _workers = 1
        if _workers > 1:
            import concurrent.futures as _cf_a
            with _cf_a.ThreadPoolExecutor(max_workers=_workers) as _ex:
                for t, a in zip(missing, _ex.map(analyze_query, missing)):
                    memo[t] = a
        else:
            for t in missing:
                try:
                    memo[t] = analyze_query(t)
                except Exception:
                    memo[t] = {"keywords": [], "entities": []}
    return [memo.get(t, {"keywords": [], "entities": []}) for t in texts]


def expand_facts_via_graph(initial_facts, kg, max_expansion_rounds=3):
    """Expand fact set by following graph connections and keyword co-occurrence (optimized)."""
    all_facts = list(initial_facts)
    seen_ids = {f.get("fact_id") for f in all_facts if f.get("fact_id")}
    from graph.expansion import batch_get_global_node_edges
    _analysis_memo = {}

    for _ in range(max_expansion_rounds):
        new_facts = []
        # Collect entities from current facts for batch graph lookup
        entity_nodes = {}
        _texts = [f.get("fact_text", "") for f in all_facts[:20]]
        _analyses = _analyze_many(_texts, _analysis_memo)
        for analysis in _analyses:
            entities = analysis.get("entities", [])
            for ent in entities:
                ent_name = ent.get("text") if isinstance(ent, dict) else str(ent)
                if not ent_name:
                    continue
                # Find global_node_id (we can batch query later; for now simple)
                conn = db.db_connect("external_graph")
                cur = conn.cursor()
                cur.execute(
                    "SELECT global_node_id FROM global_nodes WHERE canonical_name=? OR EXISTS (SELECT 1 FROM json_each(global_nodes.aliases_json) WHERE value = ?) LIMIT 1",
                    (ent_name, ent_name)
                )
                row = cur.fetchone()
                conn.close()
                if row:
                    gid = row[0]
                    entity_nodes[gid] = ent_name

        # Batch fetch edges for all entity nodes
        if entity_nodes:
            edges_by_node = batch_get_global_node_edges(list(entity_nodes.keys()))
            all_other_gids = set()
            for gid, edges in edges_by_node.items():
                for edge in edges:
                    other_gid = edge["source_node_id"] if edge["source_node_id"] != gid else edge["target_node_id"]
                    all_other_gids.add(other_gid)

            # Get canonical names for all other nodes in batch
            if all_other_gids:
                conn = db.db_connect("external_graph")
                cur = conn.cursor()
                placeholders = ",".join("?" for _ in all_other_gids)
                cur.execute(f"SELECT global_node_id, canonical_name FROM global_nodes WHERE global_node_id IN ({placeholders})",
                            list(all_other_gids))
                name_map = {row[0]: row[1] for row in cur.fetchall()}
                conn.close()

                # Retrieve facts for all other node names
                for other_gid, name in name_map.items():
                    for f in get_facts_by_keyword(name):
                        if f.get("fact_id") not in seen_ids:
                            new_facts.append(f)
                            seen_ids.add(f.get("fact_id"))

        # Collect all keywords and batch fetch facts
        all_keywords = set()
        for fact in all_facts[:20]:
            keywords = extract_keywords_from_fact(fact)
            all_keywords.update(keywords)
            for kw in keywords:
                for rel_kw, _ in get_related_keywords(kw, min_weight=0.3):
                    all_keywords.add(rel_kw)

        if all_keywords:
            from graph.expansion import batch_get_facts_by_keywords
            batch_facts = batch_get_facts_by_keywords(list(all_keywords), limit_per_keyword=20)
            for f in batch_facts:
                if f.get("fact_id") not in seen_ids:
                    new_facts.append(f)
                    seen_ids.add(f.get("fact_id"))

        if not new_facts:
            break
        all_facts.extend(new_facts)

        all_facts.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    # Content dedupe post-sort keeps the best-confidence copy of each claim
    # (converging multi-hop paths must not multiply the same fact).
    from graph.expansion import dedupe_facts_content
    return dedupe_facts_content(all_facts)[:100]


def adaptive_reasoning(query, kg=None, max_rounds=3):
    if kg is None:
        kg = KGQueryAgent()

    sub_questions = decompose_query(query)

    # Initial retrieval
    analysis = analyze_query(query)
    initial_facts = retrieve_from_graph(analysis, top_k=20)

    # Expand via graph
    expanded_facts = expand_facts_via_graph(initial_facts, kg)

    # Typed handoff check: build + validate the evidence contract before reasoning
    # consumes it. Violations log loudly and fall through to the untyped dicts —
    # a contract bug must never break answers, only flag them for repair.
    try:
        from retrieval.evidence_contract import build_contract, validate_contract
        _handoff = build_contract(query, analysis, expanded_facts)
        _problems = validate_contract(_handoff)
        if _problems:
            logger.warning("Evidence contract violations: %s", "; ".join(_problems[:5]))
            try:
                from core.metrics import inc_counter as _inc
                _inc("evidence_contract_violations_total", len(_problems))
            except Exception:
                pass
        elif config.DEBUG_VERBOSE:
            print(f"    (Evidence handoff: {len(_handoff.get('claims', []))} claims, "
                  f"{len(_handoff.get('anchors', []))} anchors)")
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"    (Evidence handoff skipped: {e})")

    # Retrieve chunks only if needed (single shared budget for both passes)
    try:
        from core.model_context import answer_budget
        _budget, _blabel = answer_budget()
    except Exception:
        _budget, _blabel = None, ""
    chunks = []
    context = build_context(expanded_facts, chunks=chunks, budget_chars=_budget, model_label=_blabel)

    if not is_sufficient(query, context):
        # Fallback to chunks
        chunks = fallback_to_chunks(query, top_k=8)
        context = build_context(expanded_facts, chunks=chunks, budget_chars=_budget, model_label=_blabel)

    # Synthesize final answer through the shared funnel (same prompt, tags,
    # budget, and token policy as every other chat path).
    if expanded_facts or chunks:
        from chat.context_builder import build_tagged_context
        from chat.synthesize import synthesize_answer
        context, ordered, _ = build_tagged_context(
            expanded_facts, chunks=chunks, budget_chars=_budget, model_label=_blabel)
        answer = synthesize_answer(query, context)
        return answer, ordered
    else:
        return "I couldn't find enough information to answer that question.", []


def orchestrate_reasoning(query, session_id=None):
    answer, verified_facts = adaptive_reasoning(query)
    if getattr(config, "ENABLE_LOGIC_LEARNING_FROM_PATHS", False):
        try:
            from logic.learn import learn_logic_from_reasoning_paths
            learn_logic_from_reasoning_paths(None, None)
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            pass
    return answer, verified_facts
