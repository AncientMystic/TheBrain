import json
from reasoning.decompose import decompose_query
from reasoning.agents import MindMapAgent, KGQueryAgent, SourceCheckerAgent, ContradictionDetectorAgent, LogicVerifierAgent
from reasoning.verify import verify_claim
from reasoning.governance import compute_confidence, store_provenance
from chat.query_analyzer import analyze_query
from chat.retriever import retrieve_from_graph, fallback_to_chunks
from chat.context_builder import build_context
from core.llm import call_model, call_model_json
from graph.graph_queries import get_related_keywords, get_facts_by_keyword, get_global_node_edges
from graph.expansion import pre_select_candidates
from core import db


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


def expand_facts_via_graph(initial_facts, kg, max_expansion_rounds=3):
    """Expand fact set by following graph connections and keyword co-occurrence (optimized)."""
    all_facts = list(initial_facts)
    seen_ids = {f.get("fact_id") for f in all_facts if f.get("fact_id")}
    from graph.expansion import batch_get_global_node_edges

    for _ in range(max_expansion_rounds):
        new_facts = []
        # Collect entities from current facts for batch graph lookup
        entity_nodes = {}
        for fact in all_facts[:20]:
            fact_text = fact.get("fact_text", "")
            analysis = analyze_query(fact_text)
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
    return all_facts[:100]


def adaptive_reasoning(query, kg=None, max_rounds=3):
    if kg is None:
        kg = KGQueryAgent()

    sub_questions = decompose_query(query)

    # Initial retrieval
    analysis = analyze_query(query)
    initial_facts = retrieve_from_graph(analysis, top_k=20)

    # Expand via graph
    expanded_facts = expand_facts_via_graph(initial_facts, kg)

    # Retrieve chunks only if needed
    chunks = []
    context = build_context(expanded_facts, chunks=chunks)

    if not is_sufficient(query, context):
        # Fallback to chunks
        chunks = fallback_to_chunks(query, top_k=8)
        context = build_context(expanded_facts, chunks=chunks)

    # Synthesize final answer
    if expanded_facts or chunks:
        facts_context = "\n".join([f"- {f.get('fact_text','')}" for f in expanded_facts[:50]])
        chunks_context = "\n".join([f"- {text[:400]}" for _, _, _, text in chunks[:8]])
        combined = "Verified facts:\n" + facts_context + "\n\nRelevant excerpts:\n" + chunks_context
        prompt = f"Using the following information, answer the user's question clearly and completely.\nQuestion: {query}\n\n{combined}\n\nAnswer:"
        answer = call_model(prompt, max_tokens=1024)
        return answer, expanded_facts
    else:
        return "I couldn't find enough information to answer that question.", []


def orchestrate_reasoning(query, session_id=None):
    answer, verified_facts = adaptive_reasoning(query)
    if getattr(config, "ENABLE_LOGIC_LEARNING_FROM_PATHS", False):
        try:
            from logic.learn import learn_logic_from_reasoning_paths
            learn_logic_from_reasoning_paths(None, None)
        except Exception:
            pass
    return answer, verified_facts
