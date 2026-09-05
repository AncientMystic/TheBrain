#!/usr/bin/env python3
"""
diagnose_all.py

Comprehensive diagnostic for TheBrain.
Checks imports, config, DBs, endpoints, retrieval, verification,
topic shift, GNN, logging, metrics, and server — without heavy processing.

Run from project root: python diagnose_all.py
"""

import sys
import time
import traceback
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

passed = []
failed = []

def check(name, func):
    """Run a check and record pass/fail."""
    print(f"\n=== {name} ===")
    try:
        func()
        passed.append(name)
        print(f"✅ PASS: {name}")
    except Exception as e:
        failed.append(name)
        print(f"❌ FAIL: {name}")
        traceback.print_exc(limit=1)

def main():
    # 1. Config import and basic attributes
    def config_check():
        import config
        assert hasattr(config, 'LLM_ENDPOINTS'), "Missing LLM_ENDPOINTS"
        assert hasattr(config, 'EMBEDDING_ENDPOINTS'), "Missing EMBEDDING_ENDPOINTS"
        assert len(config.LLM_ENDPOINTS) > 0, "No LLM endpoints"
        assert len(config.EMBEDDING_ENDPOINTS) > 0, "No embedding endpoints"
        print(f"  LLM endpoints: {len(config.LLM_ENDPOINTS)}")
        print(f"  Embedding endpoints: {len(config.EMBEDDING_ENDPOINTS)}")
        caps = getattr(config, 'LLM_ENDPOINT_CAPACITIES', [])
        print(f"  Capacities: {caps}")
        assert len(caps) >= len(config.LLM_ENDPOINTS), "Capacities mismatch"
        print("  Config OK")

    check("Config", config_check)

    # 2. Core DB connections
    def db_check():
        from core import db
        db_names = ['index', 'summaries', 'key_facts', 'embeddings', 'hypergraph',
                    'external_graph', 'ocr', 'memories', 'logic', 'reasoning',
                    'recoll_log', 'verification_standards']
        for name in db_names:
            conn = db.db_connect(name)
            conn.execute("SELECT 1")
            conn.close()
            print(f"  DB {name}: OK")
        print("  All DBs accessible")

    check("Database connections", db_check)

    # 3. LLM endpoints health (quick ping)
    def llm_health():
        import config
        from core.llm import call_model
        for i, ep in enumerate(config.LLM_ENDPOINTS):
            start = time.time()
            resp = call_model("ping", max_tokens=2, endpoint=ep)
            latency = time.time() - start
            status = "OK" if resp else "EMPTY"
            print(f"  Endpoint {i}: {status} ({latency:.2f}s)")

    check("LLM endpoint health", llm_health)

    # 4. Embedding generation
    def embedding_check():
        from core.embeddings import get_embedding
        import config as _cfgd
        emb = get_embedding("test sentence")
        assert emb is not None and len(emb) > 0, "Embedding failed"
        print(f"  Embedding dim: {len(emb)}")
        exp = int(getattr(_cfgd, "EMBEDDING_DIM", 1024))
        assert len(emb) == exp, f"Dim {len(emb)} != contract {exp} (poison risk — check BACKEND_EMBEDDINGS_MODEL)"

    check("Embedding generation", embedding_check)

    # 4b. Embedding alignment (poison prevention)
    def embedding_alignment():
        from core.embeddings import validate_embedding_config
        ok, warns = validate_embedding_config(probe=True)
        print(f"  Aligned endpoints: {len(ok)}, warnings: {len(warns)}")
        for w in warns:
            print(f"  [WARNING] {w[:300]}")

    check("Embedding alignment", embedding_alignment)

    # 5. Retrieval imports and basic functions
    def retrieval_check():
        from retrieval.features import compute_features
        from retrieval.ranking import LinearRanker, FallbackRanker
        from retrieval.orchestrator import RetrievalOrchestrator
        from retrieval.datapoint_retriever import retrieve_datapoints
        print("  Retrieval modules import OK")
        # Test linear ranker with dummy data
        ranker = LinearRanker()
        dp = {'id': 'fact:1', 'type': 'fact', 'text': 'Marie Curie discovered radium',
              'doc_hash': 'dummy', 'confidence': 0.9}
        score = ranker.score('What did Marie Curie discover?', dp, ['Marie Curie', 'radium'])
        assert isinstance(score, float), "Score not float"
        print(f"  Linear ranker score: {score:.3f}")

    check("Retrieval components", retrieval_check)

    # 6. Verification manager
    def verification_check():
        from reasoning.verification_manager import VerificationManager
        vm = VerificationManager()
        facts = [
            {'fact_text': 'Water freezes at 0C', 'canonical_value': '0C', 'confidence': 0.9},
            {'fact_text': 'Water boils at 100C', 'canonical_value': '100C', 'confidence': 0.8},
        ]
        verified = vm.verify_batch(facts)
        assert len(verified) == 2, "Verification batch failed"
        print(f"  Verified {len(verified)} facts; statuses: {[f['verification_status'] for f in verified]}")

    check("Verification Manager", verification_check)

    # 7. Topic shift model loading (if exists)
    def topic_shift_check():
        from core.topic_shift_model import TopicShiftModelDetector
        detector = TopicShiftModelDetector()
        if detector.model is not None:
            print("  Topic shift model loaded")
        else:
            print("  Topic shift model not found (fallback will be used)")

    check("Topic shift model", topic_shift_check)

    # 8. GNN module import and model existence
    def gnn_check():
        from graph.gnn_sage import SparseGraphSAGE, load_gnn_model, get_gnn_embeddings
        model = load_gnn_model()
        if model:
            print("  GNN model loaded")
        else:
            print("  GNN model not found (train with scripts/train_gnn.py)")
        emb = get_gnn_embeddings()
        if emb is not None:
            print(f"  GNN embeddings shape: {emb.shape}")
        else:
            print("  No GNN embeddings found")

    check("GNN module", gnn_check)

    # 9. Logging and metrics
    def logging_metrics_check():
        from core.logging_config import setup_logging
        from core.metrics import inc_counter, get_counter, get_all_metrics
        setup_logging()
        inc_counter("diagnostic_test")
        assert get_counter("diagnostic_test") == 1
        metrics_text = get_all_metrics()
        assert "diagnostic_test" in metrics_text
        print("  Logging/metrics OK")

    check("Logging and metrics", logging_metrics_check)

    # 10. Server app import
    def server_check():
        import server
        assert server.app is not None
        print("  Server app import OK")

    check("Server app", server_check)

    # 11. Chunk extraction function syntax (import only)
    def llm_extractor_check():
        from extraction.llm_extractor import extract_from_chunks
        print("  llm_extractor import OK")

    check("LLM extractor import", llm_extractor_check)

    # 12. Main module import
    def main_import_check():
        import main
        print("  main.py import OK")

    check("main.py import", main_import_check)

    # Summary
    print("\n" + "="*50)
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailed checks:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("All checks passed ✅")

if __name__ == "__main__":
    main()
