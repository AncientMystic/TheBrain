"""
Deep Research Coordinator.
Orchestrates agents to explore a topic and generate a comprehensive report.
"""
import uuid
import json
from core import db
from core.llm import call_model_json
from chat import analyze_query, retrieve_from_graph, fallback_to_chunks
from graph.expansion import expand_facts_via_multi_hop
from deep_research.mindmap import init_mindmap_db, add_research_node, add_research_edge, get_mindmap_text
from deep_research.report_generator import generate_report
import config

class DeepResearchCoordinator:
    def __init__(self, session_id=None):
        self.session_id = session_id or f"research_{uuid.uuid4().hex[:8]}"
        self.research_id = self.session_id
        init_mindmap_db()
        self.facts = []
        self.chunks = []

    def run(self, query):
        """Run deep research on a topic."""
        print(f"\n=== Starting deep research on: {query} ===")
        # Initial retrieval
        analysis = analyze_query(query)
        try:
            from retrieval.datapoint_retriever import retrieve_datapoints
            datapoints = retrieve_datapoints(query)
            initial_facts = [dp for dp in datapoints if dp.get("type") == "fact"]
            # Convert fact dicts to expected format if needed
        except Exception:
            initial_facts = retrieve_from_graph(analysis, top_k=50, max_depth=2)
        self.facts.extend(initial_facts)
        # Expand via multi-hop
        expanded = expand_facts_via_multi_hop(initial_facts, max_depth=config.DEEP_RESEARCH_MAX_DEPTH, max_facts=200)
        self.facts = expanded
        # Get relevant chunks
        chunks = fallback_to_chunks(query, top_k=10)
        self.chunks = chunks

        # Build mindmap from facts
        self._build_mindmap(query, self.facts)

        # Generate report
        print("Generating report...")
        report_dir = config.DEEP_RESEARCH_REPORT_DIR
        report_path = generate_report(self.research_id, self.facts, self.chunks, report_dir)
        print(f"Report saved to: {report_path}")
        return report_path


    def expand_subtopics(self, query, current_depth=0):
        """Recursively expand subtopics and generate reports for each."""
        if current_depth >= config.DEEP_RESEARCH_AUTO_SUBTOPIC_DEPTH:
            return
        subtopics = self._discover_subtopics(query)
        for subtopic in subtopics:
            print(f"  [Deep Research] Exploring subtopic: {subtopic}")
            if config.DEEP_RESEARCH_INTERACTIVE:
                user_input = input(f"Continue with '{subtopic}'? (y/n): ").strip().lower()
                if user_input != 'y':
                    continue
            self.run(subtopic)  # generate report for subtopic
            self.expand_subtopics(subtopic, current_depth + 1)

    def _discover_subtopics(self, query):
        """Use LLM and graph to suggest related subtopics."""
        prompt = f"Given the main research topic '{query}', suggest {config.DEEP_RESEARCH_MAX_SUBTOPICS} closely related subtopics. Return as a JSON array of strings."
        data = call_model_json(prompt, max_tokens=256, unwrap_list=False)
        llm_subs = data if isinstance(data, list) else []
        graph_subs = self._discover_subtopics_from_graph(query)
        # Combine and de-duplicate
        combined = []
        seen = set()
        for s in llm_subs + graph_subs:
            if s not in seen and len(s) > 0:
                combined.append(s)
                seen.add(s)
        return combined[:config.DEEP_RESEARCH_MAX_SUBTOPICS]
    def _build_mindmap(self, query, facts):
        """Create mindmap nodes/edges from facts and entities."""
        # Root node
        root_id = add_research_node(self.research_id, "topic", query, content="Main query", confidence=1.0)
        # Add fact nodes and edges in batches
        fact_nodes = []
        fact_edges = []
        for fact in facts[:100]:
            fact_text = fact.get("fact_text", "")
            confidence = fact.get("confidence", 0.5)
            node_id = add_research_node(self.research_id, "fact", fact_text[:200], content=fact_text, confidence=confidence)
            fact_nodes.append((node_id, fact_text))
            fact_edges.append((root_id, node_id, "contains"))

        # Insert edges via executemany (needs direct DB access)
        conn = db.db_connect("reasoning")
        conn.executemany("""
            INSERT INTO research_edges (research_id, source_node_id, target_node_id, relation_type)
            VALUES (?, ?, ?, ?)
        """, [(self.research_id, src, tgt, rel) for (src, tgt, rel) in fact_edges])
        conn.commit(); conn.close()

        # Entity nodes
        for fact in facts[:50]:
            canonical = fact.get("canonical_value")
            if canonical:
                node_id = add_research_node(self.research_id, "entity", canonical, content=canonical, confidence=fact.get("confidence",0.5))
                add_research_edge(self.research_id, root_id, node_id, "related_to")
