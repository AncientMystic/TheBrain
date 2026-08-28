import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
import sys, time, gc, json, traceback, sqlite3, re, os
from pathlib import Path
import numpy as np
import config
import hashlib
from core.logger import get_logger
logger = get_logger(__name__)

def validate_config():
    """Check basic requirements and optionally progress bar availability."""
    if not config.LLM_ENDPOINTS:
        print("Error: No LLM endpoints configured.")
        sys.exit(1)
    if config.USE_PROGRESS_BARS:
        try:
            import tqdm
            config.TQDM_AVAILABLE = True
        except ImportError:
            config.TQDM_AVAILABLE = False
            print("tqdm not installed; falling back to normal prints.")

from core import db
from core.progress import ProgressTracker
from core.file_utils import get_file_hash
from core.embeddings import get_embeddings_batch
from ingestion.scanner import scan_files
from ingestion.document_store import store_document, store_chunks
from ingestion.chunker import chunk_document
from extractors.registry import extract_text_from_file
from extraction.llm_extractor import extract_from_chunks
from extraction.summarizer import summarize_document
from graph.hypergraph_builder import build_hypergraph
from graph.external_graph_builder import build_external_graph
from audit.auditor import audit_all
from scripts.init_schemas import init_all
from memory import retrieve_memories, store_memory
from logic import retrieve_logic_modules, decide_logic_modules
from reasoning.orchestrator import orchestrate_reasoning
from recoll_fast import process_recoll_fast, collect_seed_keywords
from deep_research.recoll_guided_learning import run_recoll_guided_learning


def normalize_key(text):
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = ""
    return re.sub(r'\s+', ' ', text.lower()).strip()


def _safe_str_for_db(value):
    """Convert any value to string, or return empty string for None/dict/list."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def deduplicate_list(items, key_func):
    seen = {}
    for item in items:
        key = key_func(item)
        if key in seen:
            if item.get("confidence", 0) > seen[key].get("confidence", 0):
                seen[key] = item
        else:
            seen[key] = item
    return list(seen.values())


def process_file(filepath, tracker, logic_context=""):
    import numpy as np
    import json
    _t0 = time.time()
    file_hash = get_file_hash(filepath)
    if tracker.is_processed(file_hash):
        print(f"Skipping already processed: {filepath.name}")
        return False
    print(f"\n[{tracker.processed_count}/{tracker.total_files}] Processing: {filepath}")
    logger.info(f"Processing file {file_hash}", extra={'file_hash': file_hash})
    logger.info(f"Processing file {file_hash}", extra={'file_hash': file_hash})
    try:
        # Check document text cache first
        conn_cache = db.db_connect("index")
        cur_cache = conn_cache.cursor()
        cur_cache.execute("SELECT text, metadata_json FROM document_text_cache WHERE file_hash=?", (file_hash,))
        cache_row = cur_cache.fetchone()
        conn_cache.close()

        if cache_row:
            text = cache_row["text"]
            metadata = json.loads(cache_row["metadata_json"])
            result = {"text": text, "metadata": metadata, "format": Path(filepath).suffix.lstrip(".").lower()}
            file_format = result["format"]
            print("  (Using cached text)")
        else:
            result = extract_text_from_file(filepath)
            text = result["text"]
            metadata = result["metadata"]
            file_format = result["format"]
            # Save to cache
            conn_cache = db.db_connect("index")
            conn_cache.execute("INSERT OR REPLACE INTO document_text_cache (file_hash, text, metadata_json) VALUES (?,?,?)",
                               (file_hash, text, json.dumps(metadata)))
            conn_cache.commit(); conn_cache.close()

        if not text:
            print("  (No text extracted; skipping)")
            return False

        print(f"  Extracted {len(text)} chars")
        conn = db.db_connect("index")
        store_document(conn, file_hash, str(filepath), filepath.name, file_format, text, metadata,
                       ocr_used=result.get("ocr_used", False), page_count=None)
        conn.commit()
        chunks = chunk_document(text)
        print(f"  Created {len(chunks)} chunks")
        store_chunks(conn, file_hash, chunks)
        conn.commit()
        conn.close()

        print("  Generating embeddings...")
        chunk_embs = get_embeddings_batch(chunks, batch_size=config.EMBEDDING_BATCH_SIZE)
        # Compute document embedding as hyperbolic Frechet mean of chunk embeddings
        if chunk_embs:
            valid_embs = [np.array(emb, dtype=np.float32) for emb in chunk_embs if emb is not None]
            if valid_embs:
                from core.hyperbolic import exp_map, frechet_mean
                hyperbolic_points = [exp_map(emb) for emb in valid_embs]
                doc_emb_hyper = frechet_mean(hyperbolic_points)
                blob = sqlite3.Binary(doc_emb_hyper.tobytes())
                conn_emb = db.db_connect("embeddings")
                conn_emb.execute("INSERT OR REPLACE INTO document_embeddings (doc_hash, embedding, model, embedding_space) VALUES (?,?,?, 'hyperbolic')",
                                 (file_hash, blob, config.EMBEDDING_MODEL))
                conn_emb.commit(); conn_emb.close()
        print("  Extracting knowledge via LLM...")
        print("  Extracting knowledge via LLM...")
        from core.metrics import inc_counter, Timer
        inc_counter('files_processed_total')
        with Timer('extraction_duration_seconds'):
            chunk_results = extract_from_chunks(chunks, model=None, chunk_embeddings=chunk_embs, logic_context=logic_context)
        all_extracted = {"facts": [], "entities": [], "relationships": [], "people": [], "locations": [],
                         "dates": [], "events": [], "discoveries": [], "gems": []}
        for chunk_data in chunk_results:
            for key in all_extracted:
                if key in chunk_data:
                    all_extracted[key].extend(chunk_data[key])

        print("  Validating and deduplicating...")
        validated_facts = [f for f in all_extracted["facts"] if isinstance(f, dict) and f.get("source_span") and f.get("confidence",0) >= 0.5]
        all_extracted["facts"] = deduplicate_list(validated_facts, key_func=lambda f: normalize_key(f.get("fact_text","")))

        from reasoning.verification_manager import VerificationManager
        vm = VerificationManager()
        all_extracted["facts"] = vm.verify_batch(all_extracted["facts"])
        all_extracted["entities"] = deduplicate_list(all_extracted["entities"], key_func=lambda e: normalize_key(e.get("entity_name","")))
        all_extracted["people"] = deduplicate_list(all_extracted["people"], key_func=lambda p: normalize_key(p.get("person_name","")))
        all_extracted["locations"] = deduplicate_list(all_extracted["locations"], key_func=lambda l: normalize_key(l.get("location_name","")))
        all_extracted["dates"] = deduplicate_list(all_extracted["dates"], key_func=lambda d: normalize_key(d.get("date_text","")))
        all_extracted["events"] = deduplicate_list(all_extracted["events"], key_func=lambda e: normalize_key(e.get("event_name","")))
        all_extracted["discoveries"] = deduplicate_list(all_extracted["discoveries"], key_func=lambda d: normalize_key(d.get("discovery_name","")))
        all_extracted["gems"] = deduplicate_list(all_extracted["gems"], key_func=lambda g: normalize_key(g.get("gem_text","")))

        # Collect gate training data (label based on verified facts)
        if getattr(config, "USE_PRIME_EVEN_GATE", False):
            try:
                from core.spectral import compute_spectral_features
                import numpy as np
                if chunk_embs:
                    emb_matrix = np.array([np.array(e, dtype=np.float32) for e in chunk_embs if e is not None])
                    if emb_matrix.shape[0] > 0:
                        feat = compute_spectral_features(emb_matrix, top_k=22)
                        label = 0
                        for fact in all_extracted.get("facts", []):
                            if fact.get("verification_status") in ("verified", "partially_verified"):
                                label = 1
                                break
                        conn_gate = db.db_connect("key_facts")
                        conn_gate.execute(
                            "INSERT INTO gate_training_data (chunk_hash, features, label) VALUES (?,?,?)",
                            (file_hash, sqlite3.Binary(feat.tobytes()), label)
                        )
                        conn_gate.commit()
                        conn_gate.close()
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"    (Gate training data collection error: {e})")

        # Collect verification gate training data
        if getattr(config, "USE_GATED_VERIFICATION", False):
            try:
                import numpy as np
                import json
                from core.spectral import compute_spectral_features
                from core.embeddings import get_embedding
                from reasoning.verification_gate import VerificationGate
                # Use the first fact with verification layers as representative
                for fact in all_extracted.get("facts", []):
                    if not isinstance(fact, dict):
                        continue
                    fact_text = fact.get("fact_text", "")
                    emb = get_embedding(fact_text)
                    if emb is None:
                        continue
                    features = compute_spectral_features(np.array([emb], dtype=np.float32))
                    layers = fact.get("verification_layers", [])
                    if not layers:
                        continue
                    # Build a dict of verifier_name -> success (1/0)
                    label_dict = {}
                    for v in layers:
                        layer_name = v.get("layer")
                        if layer_name:
                            label_dict[layer_name] = 1 if v.get("verified") else 0
                    if not label_dict:
                        continue
                    # Store features and labels (fact_id is NULL for now)
                    conn_gate = db.db_connect("reasoning")
                    conn_gate.execute(
                        "INSERT INTO verification_gate_training_data (fact_id, features, labels) VALUES (?,?,?)",
                        (fact.get("fact_id"), sqlite3.Binary(features.tobytes()), json.dumps(label_dict))
                    )
                    conn_gate.commit()
                    conn_gate.close()
                    break  # only one per document for now
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"    (Verification gate training data collection error: {e})")

        print("  Storing key facts...")
        conn_facts = db.db_connect("key_facts")
        cur_facts = conn_facts.cursor()
        conn_index = db.db_connect("index")
        cur_index = conn_index.cursor()
        fact_rows = []
        source_rows = []
        entity_index_rows = []
        # Rerank extracted facts against document name before storage
        if getattr(config, "RERANKER_ENABLED", True):
            try:
                from retrieval.ingest_ranker import rank_extracted_items
                all_extracted["facts"] = rank_extracted_items(filepath.name, all_extracted["facts"], "fact_text")
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"    (Ingest rank error: {e})")

        # Compute verification flags (advisory)
        from reasoning.verify import verify_symstep
        for i, fact in enumerate(all_extracted["facts"]):
            prior = all_extracted["facts"][:i]
            sym_contradiction = 0
            formal_repr = None
            rcot_verified = 0

            # SymStep advisory
            try:
                sym_ok = verify_symstep(fact, prior)
                if not sym_ok:
                    sym_contradiction = 1
            except Exception:
                pass

            # VeriCoT formal representation (if possible)
            try:
                from reasoning.verify import extract_triple_from_text
                triple = extract_triple_from_text(fact.get("fact_text", ""))
                if triple and all(k in triple for k in ("subject", "predicate", "object")):
                    formal_repr = json.dumps(triple)
            except Exception:
                pass

            # R-CoT verification for lower confidence facts
            if fact.get("confidence", 0) < 0.7:
                try:
                    from reasoning.verify import verify_rcot
                    if verify_rcot(fact.get("fact_text", ""), None):
                        rcot_verified = 1
                except Exception:
                    pass

            fact["_sym_contradiction"] = sym_contradiction
            fact["_formal_repr"] = formal_repr
            fact["_rcot_verified"] = rcot_verified

        if getattr(config, "ENABLE_BATCH_VERIFICATION", False):
            try:
                from processing.verification_batch import verify_batch
                all_extracted["facts"] = verify_batch(all_extracted["facts"])
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"    (Batch verification error: {e})")

        for fact in all_extracted["facts"]:
            if not isinstance(fact, dict):
                continue
            fact_text = fact.get("fact_text")
            if not fact_text or not isinstance(fact_text, str) or not fact_text.strip():
                continue
            # Ensure fact_type is string
            fact_type = fact.get("fact_type")
            if not isinstance(fact_type, str):
                fact_type = "other"
            fact_row = (
                file_hash, filepath.name, fact_type,
                fact_text, _safe_str_for_db(fact.get("canonical_value")),
                _safe_str_for_db(fact.get("source_span")), fact.get("confidence_final", fact.get("confidence", 0.0)), 0,
                fact.get("_sym_contradiction", 0), _safe_str_for_db(fact.get("_formal_repr", "")), fact.get("_rcot_verified", 0)
            )
            fact_rows.append(fact_row)

            # Store source span as quote if enabled
            if getattr(config, "ENABLE_QUOTE_STORAGE", True) and fact.get("source_span"):
                span = fact.get("source_span")
                cur_facts.execute("""
                    INSERT INTO quotes (doc_hash, chunk_id, quote_text, canonical_value, confidence)
                    VALUES (?, NULL, ?, ?, ?)
                """, (file_hash, span, fact.get("canonical_value", ""), fact.get("confidence", 0.0)))

        if fact_rows:
            cur_facts.executemany(
                "INSERT INTO key_facts (doc_hash, doc_name, fact_type, fact_text, canonical_value, source_span, confidence, verified, symstep_contradiction, formal_representation, rcot_verified) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                fact_rows,
            )

        # Retrieve last inserted IDs
        cur_facts.execute("SELECT fact_id, source_span, canonical_value FROM key_facts WHERE doc_hash=? ORDER BY fact_id DESC LIMIT ?",
                          (file_hash, len(fact_rows)))
        inserted = cur_facts.fetchall()[::-1]

        for fact, inserted_row in zip(all_extracted["facts"], inserted):
            fact_id = inserted_row["fact_id"]
            span = fact.get("source_span","")
            cur_index.execute("SELECT chunk_id FROM document_chunks WHERE doc_hash=? AND chunk_text LIKE ? LIMIT 1", (file_hash, f"%{span}%"))
            row = cur_index.fetchone()
            chunk_id = row[0] if row else None
            if chunk_id:
                source_rows.append((fact_id, file_hash, chunk_id, span, span))
            if fact.get("canonical_value"):
                try:
                    from core.fact_normalizer import normalize_name
                    norm = normalize_name(fact["canonical_value"])
                    entity_index_rows.append((fact_id, fact["canonical_value"], norm))
                except ImportError:
                    pass

        if source_rows:
            cur_facts.executemany(
                "INSERT INTO fact_sources (fact_id, doc_hash, chunk_id, evidence_span, exact_quote) VALUES (?,?,?,?,?)",
                source_rows,
            )
        if entity_index_rows:
            cur_facts.executemany(
                "INSERT OR IGNORE INTO entity_fact_index (fact_id, entity_name, normalized_name) VALUES (?, ?, ?)",
                entity_index_rows,
            )
        # Store all categories
        for entity in all_extracted.get("entities", []):
            _store_entity(conn_facts, file_hash, filepath.name, entity)
        for person in all_extracted.get("people", []):
            _store_person(conn_facts, file_hash, filepath.name, person)
        for location in all_extracted.get("locations", []):
            _store_location(conn_facts, file_hash, filepath.name, location)
        for date in all_extracted.get("dates", []):
            _store_date(conn_facts, file_hash, filepath.name, date)
        for event in all_extracted.get("events", []):
            _store_event(conn_facts, file_hash, filepath.name, event)
        for discovery in all_extracted.get("discoveries", []):
            _store_discovery(conn_facts, file_hash, filepath.name, discovery)
        for gem in all_extracted.get("gems", []):
            _store_gem(conn_facts, file_hash, filepath.name, gem)
        conn_facts.commit(); conn_facts.close(); conn_index.close()

        # Statistical keyword extraction (if enabled)
        if getattr(config, "ENABLE_STATISTICAL_KEYWORDS", True):
            try:
                from extraction.keyword_extractor import extract_rake_phrases
                stat_keywords = extract_rake_phrases(text, max_phrases=10)
                if stat_keywords:
                    # Store in keyword_topic_edges or logic_keywords
                    conn_kw = db.db_connect("external_graph")
                    cur_kw = conn_kw.cursor()
                    for kw in stat_keywords:
                        cur_kw.execute("""
                            INSERT OR IGNORE INTO keyword_topic_edges (keyword, topic, weight)
                            VALUES (?, ?, 0.5)
                        """, (kw, filepath.name))
                    conn_kw.commit(); conn_kw.close()
                    print(f"  Extracted {len(stat_keywords)} statistical keywords")
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"    (Statistical keyword extraction error: {e})")

        print("  Building hypergraph...")
        with Timer('graph_build_duration_seconds'):
            build_hypergraph(file_hash, all_extracted, {})
            print("  Building external graph...")
            build_external_graph(file_hash, all_extracted, {})
        print("  Generating summary...")
        summary, key_points = summarize_document(chunks)
        conn_summ = db.db_connect("summaries")
        conn_summ.execute("INSERT OR REPLACE INTO doc_summaries (doc_hash, doc_name, summary, key_points_json, verification_status) VALUES (?,?,?,?,'unverified')",
                          (file_hash, filepath.name, summary, json.dumps(key_points)))
        conn_summ.commit(); conn_summ.close()
        tracker.mark_processed(file_hash)
        elapsed = time.time() - _t0
        print(f"  Done processing {filepath.name} in {elapsed:.2f}s")
        return True
    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")
        traceback.print_exc()
        # Close all DB connections before marking error to avoid lock
        try:
            from core import db as db_module
            db_module.close_all_connections()
        except Exception:
            pass
        tracker.mark_error(file_hash, stage="processing")
        return False


# Helper functions for storing extracted categories (moved here to avoid circular import)
def _store_entity(conn, doc_hash, file_name, entity):
    conn.execute("""INSERT INTO entities (doc_hash, entity_type, entity_name, normalized_name, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, _safe_str_for_db(entity.get("entity_type", "OTHER")), _safe_str_for_db(entity.get("entity_name", "")),
                  _safe_str_for_db(entity.get("normalized_name", "")), _safe_str_for_db(entity.get("source_span", "")), entity.get("confidence", 0.0)))

def _store_person(conn, doc_hash, file_name, person):
    conn.execute("""INSERT INTO people (doc_hash, person_name, normalized_name, role, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, _safe_str_for_db(person.get("person_name", "")), _safe_str_for_db(person.get("normalized_name", "")),
                  _safe_str_for_db(person.get("role", "")), _safe_str_for_db(person.get("source_span", "")), person.get("confidence", 0.0)))

def _store_location(conn, doc_hash, file_name, location):
    conn.execute("""INSERT INTO locations (doc_hash, location_name, normalized_place, location_type, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, _safe_str_for_db(location.get("location_name", "")), _safe_str_for_db(location.get("normalized_place", "")),
                  _safe_str_for_db(location.get("location_type", "")), _safe_str_for_db(location.get("source_span", "")), location.get("confidence", 0.0)))

def _store_date(conn, doc_hash, file_name, date):
    conn.execute("""INSERT INTO dates (doc_hash, date_text, normalized_date, date_type, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, _safe_str_for_db(date.get("date_text", "")), _safe_str_for_db(date.get("normalized_date", "")),
                  _safe_str_for_db(date.get("date_type", "")), _safe_str_for_db(date.get("source_span", "")), date.get("confidence", 0.0)))

def _store_event(conn, doc_hash, file_name, event):
    conn.execute("""INSERT INTO events (doc_hash, event_name, normalized_name, event_date, event_type,
                                        description, significance, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                 (doc_hash, _safe_str_for_db(event.get("event_name", "")), _safe_str_for_db(event.get("normalized_name", "")),
                  _safe_str_for_db(event.get("event_date", "")), _safe_str_for_db(event.get("event_type", "")), _safe_str_for_db(event.get("description", "")),
                  _safe_str_for_db(event.get("significance", "")), _safe_str_for_db(event.get("source_span", "")), event.get("confidence", 0.0)))

def _store_discovery(conn, doc_hash, file_name, discovery):
    conn.execute("""INSERT INTO discoveries (doc_hash, discovery_name, normalized_name, description,
                                             date, significance, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                 (doc_hash, _safe_str_for_db(discovery.get("discovery_name", "")), _safe_str_for_db(discovery.get("normalized_name", "")),
                  _safe_str_for_db(discovery.get("description", "")), _safe_str_for_db(discovery.get("date", "")), _safe_str_for_db(discovery.get("significance", "")),
                  _safe_str_for_db(discovery.get("source_span", "")), discovery.get("confidence", 0.0)))

def _store_gem(conn, doc_hash, file_name, gem):
    conn.execute("""INSERT INTO gems (doc_hash, gem_text, category, importance, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, _safe_str_for_db(gem.get("gem_text", "")), _safe_str_for_db(gem.get("category", "")),
                  gem.get("importance", 0.0), _safe_str_for_db(gem.get("source_span", "")), gem.get("confidence", 0.0)))


def review_contradictions():
    """
    Print unresolved contradictions from the review queue.
    """
    conn = db.db_connect("reasoning")
    cur = conn.cursor()
    cur.execute("""
        SELECT id, status, resolved_by, resolved_at, triple_a_id, triple_b_id, details
        FROM contradiction_log
        WHERE status = 'review_needed'
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No contradictions awaiting review.")
        return

    print(f"Found {len(rows)} contradictions in review queue:")
    for row in rows:
        details = row["details"] or row["resolved_by"] or ""
        print(f"  ID {row['id']}: {details}")
    print("\nUse --audit to run a new audit or manually review the database.")


def promote_verified_file(file_hash, file_name, source_file=None):
    """Mark a file as verified source and insert its key facts into standards."""
    from core import db
    import json
    from core.fact_normalizer import normalize_name

    # Ensure tracking table exists
    conn_track = db.db_connect("verification_standards")
    cur_track = conn_track.cursor()
    cur_track.execute("""
        CREATE TABLE IF NOT EXISTS verification_promotions (
            doc_hash TEXT PRIMARY KEY,
            promoted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT,
            standards_inserted INTEGER DEFAULT 0
        )
    """)
    conn_track.commit()
    cur_track.execute("SELECT 1 FROM verification_promotions WHERE doc_hash=?", (file_hash,))
    already = cur_track.fetchone()
    conn_track.close()
    if already:
        print(f"    Already promoted {file_name}; skipping.")
        return

    # Mark document as verified source
    conn_idx = db.db_connect("index")
    conn_idx.execute("UPDATE documents SET is_verified_source=1 WHERE file_hash=?", (file_hash,))
    conn_idx.commit(); conn_idx.close()

    # Socratic/PSYOP assessment
    socratic_assessment = {
        "source_hierarchy_level": 0,
        "psych_score_total": 0,
        "data_model_policy": "unknown",
        "enforcement_vector": None,
        "intentionality_triad": {},
        "lived_experience_cluster": False,
        "funding_gatekeeping_flags": {},
        "summary": "Socratic assessment failed for verified source."
    }
    if source_file is not None:
        # Check cache first
        conn_vs = db.db_connect("verification_standards")
        cur_vs = conn_vs.cursor()
        cur_vs.execute("""
            SELECT socratic_assessment_json
            FROM verified_standard_sources
            WHERE source_doc_hash=?
        """, (file_hash,))
        cached = cur_vs.fetchone()
        conn_vs.close()
        if cached and cached[0]:
            try:
                socratic_assessment = json.loads(cached[0])
                print("    (Socratic assessment loaded from cache)")
            except Exception:
                pass
        else:
            try:
                from reasoning.socratic_scorer import score_document
                from extractors.registry import extract_text_from_file
                _vtext = extract_text_from_file(source_file).get("text", "")
                socratic_assessment = score_document(source_file.name, _vtext)
                print("    (Socratic assessment complete)")
                # Store cache
                conn_vs = db.db_connect("verification_standards")
                conn_vs.execute("""
                    INSERT OR REPLACE INTO verified_standard_sources
                    (source_doc_hash, file_path, title, source_hierarchy_level, socratic_assessment_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (file_hash, str(source_file), source_file.name,
                      socratic_assessment.get("source_hierarchy_level", 0),
                      json.dumps(socratic_assessment)))
                conn_vs.commit(); conn_vs.close()
            except Exception as e:
                print(f"    (Socratic assessment error: {e})")

    # Retrieve facts for this file
    conn_kf = db.db_connect("key_facts")
    cur_kf = conn_kf.cursor()
    cur_kf.execute("SELECT fact_text, canonical_value, confidence FROM key_facts WHERE doc_hash=?", (file_hash,))
    facts = cur_kf.fetchall()
    conn_kf.close()

    conn_std = db.db_connect("verification_standards")
    cur_std = conn_std.cursor()
    for fact_text, canonical_value, confidence in facts:
        subject = canonical_value or fact_text
        predicate = "has_fact"
        obj = fact_text
        standard_id = f"vf-{file_hash}-{normalize_name(fact_text)[:20]}"
        cur_std.execute("""
            INSERT OR REPLACE INTO verified_standards
            (standard_id, statement, subject, predicate, object, negation,
             truth_status, source_type, source_doc_hash, priority, confidence,
             verified_by, socratic_assessment_json)
            VALUES (?, ?, ?, ?, ?, 0, 'verified_true', 'verified_folder', ?, 1, ?, 'verified_folder', ?)
        """, (standard_id, fact_text, subject, predicate, obj, file_hash,
              confidence, json.dumps(socratic_assessment)))

        conn_kf = db.db_connect("key_facts")
        conn_kf.execute("UPDATE key_facts SET verification_status='verified_true', verified_by='verified_folder' WHERE fact_text=? AND doc_hash=?",
                        (fact_text, file_hash))
        conn_kf.commit(); conn_kf.close()

    conn_std.commit(); conn_std.close()
    print("    (Inserted extracted facts as verified standards with Socratic metadata)")

    try:
        from scripts.attach_supporting_evidence import attach_supporting_evidence_to_standards
        attach_supporting_evidence_to_standards()
    except Exception as e:
        if config.DEBUG_VERBOSE:
            print(f"    (Supporting evidence extraction error: {e})")

    # Record promotion
    conn_promo = db.db_connect("verification_standards")
    conn_promo.execute("""
        INSERT OR REPLACE INTO verification_promotions (doc_hash, promoted_at, source_type, standards_inserted)
        VALUES (?, CURRENT_TIMESTAMP, 'verified_folder', ?)
    """, (file_hash, len(facts)))
    conn_promo.commit(); conn_promo.close()
def update_status(message):
    """Update the current terminal status line in place."""
    print(f"\r{message}", end="", flush=True)


def main():
    if "--debug" in sys.argv:
        config.DEBUG_VERBOSE = True

    server_mode = "--server" in sys.argv
    guided = "--guided-learning" in sys.argv
    chat_mode = "--chat" in sys.argv
    audit_mode = "--audit" in sys.argv
    debug_retrieval = "--debug-retrieval" in sys.argv
    review_contradictions_mode = "--review-contradictions" in sys.argv
    logic_mode = "--logic" in sys.argv
    reasoning_mode = "--reasoning" in sys.argv
    deep_research = "--deep-research" in sys.argv
    recoll_mode = "--recoll" in sys.argv
    recoll_fast = "--recoll-fast" in sys.argv
    build_recoll_index = "--build-recoll-index" in sys.argv
    train_gnn_flag = "--train-gnn" in sys.argv
    maintenance_mode = "--maintenance" in sys.argv
    interactive = "--interactive" in sys.argv
    input_path = None
    if "--input" in sys.argv:
        idx = sys.argv.index("--input") + 1
        if idx < len(sys.argv):
            input_path = sys.argv[idx]

    verified_flag = "--verified" in sys.argv
    admin_facts = []
    if "--fact" in sys.argv:
        i = 0
        while i < len(sys.argv):
            if sys.argv[i] == "--fact" and i + 1 < len(sys.argv):
                admin_facts.append(sys.argv[i + 1])
                i += 2
            else:
                i += 1

    recoll_query = None
    if "--recoll-query" in sys.argv:
        idx = sys.argv.index("--recoll-query") + 1
        if idx < len(sys.argv):
            recoll_query = sys.argv[idx]

    recoll_limit = None
    if "--recoll-limit" in sys.argv:
        idx = sys.argv.index("--recoll-limit") + 1
        if idx < len(sys.argv):
            try:
                recoll_limit = int(sys.argv[idx])
            except ValueError:
                print("Invalid --recoll-limit value, using default.")

    preview_chars = None
    if "--preview-chars" in sys.argv:
        idx = sys.argv.index("--preview-chars") + 1
        if idx < len(sys.argv):
            try:
                preview_chars = int(sys.argv[idx])
            except ValueError:
                print("Invalid --preview-chars value, using default.")

    recoll_model = None
    if "--recoll-model" in sys.argv:
        idx = sys.argv.index("--recoll-model") + 1
        if idx < len(sys.argv):
            recoll_model = sys.argv[idx]

    dry_run = "--dry-run" in sys.argv
    limit_files = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit") + 1
        if idx < len(sys.argv):
            try:
                limit_files = int(sys.argv[idx])
            except ValueError:
                print("Invalid --limit value, ignoring.")

    validate_config()
    init_all()

    # Train GNN if requested
    if train_gnn_flag:
        print("Training GNN embeddings...")
        from graph.gnn import train_gnn
        emb = train_gnn()
        if emb is not None:
            print(f"GNN embeddings shape: {emb.shape}")
        else:
            print("GNN training returned no embeddings.")
        return

    # Handle recoll-fast first (only if flag present)
    if recoll_fast:
        from recoll_fast import process_recoll_fast, collect_seed_keywords
        if recoll_query:
            process_recoll_fast(recoll_query, max_results=recoll_limit,
                                preview_chars=preview_chars, model=recoll_model)
        elif interactive:
            print("Recoll Fast interactive mode. Type 'exit' to quit.")
            while True:
                try:
                    q = input("Recoll query> ").strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if q.lower() in ("exit", "quit"):
                    break
                if not q:
                    continue
                process_recoll_fast(q, max_results=recoll_limit,
                                    preview_chars=preview_chars, model=recoll_model)
        else:
            # Automatic mode: collect seed keywords
            print("Recoll Fast automatic mode - collecting seed keywords...")
            seeds = collect_seed_keywords(limit=config.RECOLL_AUTO_KEYWORD_LIMIT if hasattr(config, 'RECOLL_AUTO_KEYWORD_LIMIT') else 20)
            if not seeds:
                print("No seed keywords found. Use --recoll-query or build knowledge first.")
            else:
                print(f"Processing {len(seeds)} keywords automatically...")
                for kw in seeds:
                    print(f"\n=== Processing keyword: {kw} ===")
                    process_recoll_fast(kw, max_results=recoll_limit,
                                        preview_chars=preview_chars, model=recoll_model)
        return

    if build_recoll_index:
        if not input_path:
            print("Please provide --input <path> for building Recoll index.")
            return
        try:
            import recoll
            files = scan_files(input_path)
            print(f"Indexing {len(files)} files with Recoll...")
            recoll_db = recoll.connect(writable=True)
            for i, f in enumerate(files):
                print(f"  Indexing {i+1}/{len(files)}: {f.name}")
                try:
                    result = extract_text_from_file(f)
                    text = result.get("text", "")
                    if not text:
                        continue
                    doc = recoll.Doc()
                    doc.url = f.as_uri()
                    doc.title = f.stem
                    doc.mimetype = result.get("format", "text/plain")
                    doc.text = text
                    file_hash = get_file_hash(str(f))
                    udi = f"thebrain:{file_hash}"
                    if recoll_db.needUpdate(udi, file_hash):
                        recoll_db.addOrUpdate(udi, doc)
                except Exception as e:
                    print(f"    Recoll indexing error: {e}")
            recoll_db.close()
            print("Recoll index build complete.")
        except ImportError as e:
            print(f"Recoll not available: {e}")
        return

    if maintenance_mode:
        from core.maintenance import run_maintenance_once, start_background_maintenance
        interval = 3600
        if "--interval" in sys.argv:
            idx = sys.argv.index("--interval") + 1
            if idx < len(sys.argv):
                try:
                    interval = int(sys.argv[idx])
                except ValueError:
                    pass
        start_background_maintenance(interval_seconds=interval)
        print(f"Maintenance mode started (interval {interval}s). Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Exiting maintenance mode.")
        return

    if server_mode:
        import uvicorn
        from server import app as server_app
        print(f"Starting OpenAI-compatible server on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
        uvicorn.run(server_app, host=config.SERVER_HOST, port=config.SERVER_PORT)
        return

    if review_contradictions_mode:
        review_contradictions()
        return

    if audit_mode:
        audit_all()
        return

    if chat_mode or deep_research:
        from chat import analyze_query, retrieve_from_graph, fallback_to_chunks, build_context, generate_answer
        from chat.conversation import add_message, get_conversation_context
        from chat.conversation_state import ConversationTopicState
        from deep_research.coordinator import DeepResearchCoordinator

        print("Chat mode. Type 'exit' to quit. Add --deep-research to enable autonomous research.")
        session_id = f"cli_{int(time.time())}"
        topic_state = ConversationTopicState()
        while True:
            try:
                query = input("You: ")
                if query.lower() in ["exit", "quit"]:
                    break
                if query.startswith("remember:"):
                    mem_content = query[len("remember:"):].strip()
                    store_memory(session_id, mem_content, memory_type="user_note")
                    print("Memory stored.")
                    continue

                add_message(session_id, "user", query)
                conversation_history = get_conversation_context(session_id)

                if deep_research:
                    coordinator = DeepResearchCoordinator(session_id)
                    report_path = coordinator.run(query)
                    answer = f"Deep research report generated: {report_path}"
                elif reasoning_mode:
                    answer, _ = orchestrate_reasoning(query)
                else:
                    logic_ids = decide_logic_modules(query, context=query[:1000])
                    logic_context = ""
                    if logic_ids:
                        conn = db.db_connect("logic")
                        cur = conn.cursor()
                        for lid in logic_ids:
                            cur.execute("SELECT name, category, summary, content FROM logic_modules WHERE logic_id=?", (lid,))
                            row = cur.fetchone()
                            if row:
                                logic_context += f"[Logic: {row[0]} ({row[1]})]\n{row[2]}\n{row[3]}\n\n"
                        conn.close()

                    memories = retrieve_memories(query, top_k=5, session_id=session_id)
                    memory_text = "\n".join([f"[Memory] {m[2]}" for m in memories])

                    analysis = analyze_query(query)
                    if debug_retrieval:
                        print("\n[DEBUG-RETRIEVAL] Query:", query)

                    detail_mode = analysis.get("intent") == "detail"
                    chunk_top_k = max(getattr(config, "CHAT_TOP_K_CHUNKS", 10), 15) if detail_mode else getattr(config, "CHAT_TOP_K_CHUNKS", 10)
                    extra_terms = topic_state.get_active_terms()

                    if getattr(config, 'USE_MULTI_STAGE_RETRIEVAL', True):
                        from retrieval.orchestrator import RetrievalOrchestrator
                        orchestrator = RetrievalOrchestrator()
                        datapoints = orchestrator.retrieve(query, analysis)
                        facts = [dp for dp in datapoints if dp.get('type') == 'fact']
                        chunks = []
                        for dp in datapoints:
                            if dp.get('type') == 'chunk_ref':
                                chunks.append((0, dp.get('chunk_id'), dp.get('doc_hash'), dp.get('text', '')))
                        context = build_context(facts, chunks=chunks, conversation_history=conversation_history, detail_mode=detail_mode)
                    else:
                        try:
                            from retrieval.datapoint_retriever import retrieve_datapoints, get_chunks_for_datapoints

                            datapoints = retrieve_datapoints(query, extra_terms=extra_terms)
                            if datapoints:
                                selected_dps = datapoints[:getattr(config, "MAX_SELECTED_NODES", 15)]

                                # Always pull document/summary datapoints, even if they rank below top-15
                                doc_dps = [dp for dp in datapoints if dp.get("type") in ("document", "summary")][:10]
                                for dp in doc_dps:
                                    if dp not in selected_dps:
                                        selected_dps.insert(0, dp)

                                chunks = get_chunks_for_datapoints(selected_dps)
                                facts = [dp for dp in selected_dps if dp.get("type") == "fact"]

                                doc_lines = []
                                for dp in doc_dps:
                                    if dp.get("type") == "document":
                                        doc_lines.append(f"[Document] {dp.get('text')}")
                                    elif dp.get("type") == "summary":
                                        doc_lines.append(f"[Summary] {dp.get('text')}")
                                doc_context = "\n".join(doc_lines) if doc_lines else ""

                                base_context = build_context(facts, chunks=chunks, conversation_history=conversation_history, detail_mode=True)
                                context = (doc_context + "\n\n" + base_context) if doc_context else base_context
                            else:
                                facts = retrieve_from_graph(analysis, top_k=50, max_depth=2, debug=debug_retrieval)
                                chunks = fallback_to_chunks(query, top_k=chunk_top_k, debug=debug_retrieval)
                                context = build_context(facts, chunks=chunks, conversation_history=conversation_history, detail_mode=detail_mode)
                        except Exception as e:
                            if debug_retrieval:
                                print(f"    (Map retrieval error: {e})")
                            facts = retrieve_from_graph(analysis, top_k=50, max_depth=2, debug=debug_retrieval)
                            chunks = fallback_to_chunks(query, top_k=chunk_top_k, debug=debug_retrieval)
                            context = build_context(facts, chunks=chunks, conversation_history=conversation_history, detail_mode=detail_mode)

                    if logic_context:
                        context = logic_context + "\n\n" + context
                    if memory_text:
                        context = memory_text + "\n\n" + context

                    if getattr(config, "USE_VERIFIED_CHAT", True):
                        from chat.give_chat import generate_answer_verified
                        answer = generate_answer_verified(query, conversation_history=conversation_history)
                    else:
                        answer = generate_answer(query, context)
                    topic_state.update(query, answer)

                if getattr(config, "USE_HYPERBOLIC_MEMORY", True):
                    from memory.hyperbolic_memory import update_session_centroid
                    update_session_centroid(session_id, query, answer)
                add_message(session_id, "assistant", answer)
                print(f"Assistant:\n{answer}\n---")
            except KeyboardInterrupt:
                print("\nExiting chat.")
                break
        return

    if logic_mode and input_path and not guided:
        from logic.learn import learn_logic_from_file
        input_path = Path(input_path)
        files = [input_path] if input_path.is_file() else scan_files(input_path)
        for f in files:
            print(f"Learning logic from {f.name}...")
            learn_logic_from_file(f)
        print("Logic learning complete.")
        return

    if guided and recoll_mode:
        print("Starting Recoll-guided learning mode.")
        tracker = ProgressTracker()
        tracker.total_files = 0
        tracker.processed_count = 0
        _recoll_max_rounds = config.RECOLL_MAX_ROUNDS
        if "--recoll-max-rounds" in sys.argv:
            _idx = sys.argv.index("--recoll-max-rounds") + 1
            if _idx < len(sys.argv):
                _recoll_max_rounds = int(sys.argv[_idx])
        run_recoll_guided_learning(process_file, tracker, max_rounds=_recoll_max_rounds,
                                   interactive=interactive)
        return

    if guided:
        # Process admin claims first (if any)
        if admin_facts:
            print(f"Processing {len(admin_facts)} admin claim(s)...")
            from core.llm import call_model_json
            from core.fact_normalizer import normalize_name
            conn_std = db.db_connect("verification_standards")
            cur_std = conn_std.cursor()
            conn_kf = db.db_connect("key_facts")
            cur_kf = conn_kf.cursor()
            for idx, claim in enumerate(admin_facts):
                # Use LLM to format into subject/predicate/object
                prompt = f"""You are a fact formatter, not a fact checker.
The following claim is TRUE by admin definition. Do NOT evaluate or dispute it.
Format it into JSON with keys: statement, subject, predicate, object, negation (0 or 1).
Claim: {claim}
Return only JSON."""
                data = call_model_json(prompt, max_tokens=256)
                if not isinstance(data, dict):
                    data = {}
                statement = claim  # always preserve the original admin claim exactly
                subject = data.get("subject") or claim
                predicate = data.get("predicate") or "has_truth_value"
                obj = data.get("object") or "true"
                negation = int(data.get("negation", 0) or 0)
                claim_hash = hashlib.sha1(statement.encode("utf-8")).hexdigest()[:16]
                standard_id = f"admin-{claim_hash}"
                admin_doc_hash = f"admin-claim-{claim_hash}"

                admin_socratic = {
                    "source_hierarchy_level": 0,
                    "psych_score_total": 0,
                    "data_model_policy": "data",
                    "enforcement_vector": None,
                    "intentionality_triad": {},
                    "lived_experience_cluster": False,
                    "funding_gatekeeping_flags": {},
                    "summary": "Admin claim: indisputable reference fact."
                }
                cur_std.execute("""
                    INSERT OR REPLACE INTO verified_standards
                    (standard_id, statement, subject, predicate, object, negation,
                     truth_status, source_type, source_doc_hash, priority, verified_by,
                     source_hierarchy_level, socratic_assessment_json)
                    VALUES (?, ?, ?, ?, ?, ?, 'admin_claim', 'admin_claim', NULL, 0, 'admin_claim', 0, ?)
                """, (standard_id, statement, subject, predicate, obj, negation,
                      json.dumps(admin_socratic)))

                # Remove any previous admin key fact with same doc_hash, then insert fresh.
                cur_kf.execute("DELETE FROM key_facts WHERE doc_hash=?", (admin_doc_hash,))
                cur_kf.execute("""
                    INSERT INTO key_facts (doc_hash, doc_name, fact_type, fact_text, canonical_value,
                                           source_span, confidence, verification_status, verified_by, negation)
                    VALUES (?, ?, 'admin_claim', ?, ?, ?, 1.0, 'admin_claim', 'admin_claim', ?)
                """, (admin_doc_hash, "admin_claim", statement, subject, predicate, negation))
            conn_std.commit(); conn_std.close()
            conn_kf.commit(); conn_kf.close()
            print("Admin claims stored.")

        if not input_path:
            if admin_facts:
                print("No input folder specified, but admin claims were processed.")
                return
            print("Please provide --input <path> for guided learning.")
            return
        input_path = Path(input_path)
        files = scan_files(input_path)
        total = len(files)
        tracker = ProgressTracker()
        tracker.total_files = total
        tracker.processed_count = 0
        try:
            if getattr(config, "PARALLEL_PROCESSING_ENABLED", False):
                import concurrent.futures
                from threading import Lock
                tracker_lock = Lock()
                total_files = len(files)
                processed_count = 0

                def process_one(f):
                    nonlocal processed_count
                    logic_context = ""
                    if logic_mode:
                        try:
                            result = extract_text_from_file(f)
                            first_text = result["text"][:1000]
                            logic_ids = decide_logic_modules(first_text, context=first_text)
                            if logic_ids:
                                conn = db.db_connect("logic")
                                cur = conn.cursor()
                                for lid in logic_ids:
                                    cur.execute("SELECT name, category, summary, content FROM logic_modules WHERE logic_id=?", (lid,))
                                    row = cur.fetchone()
                                    if row:
                                        logic_context += f"[Logic: {row[0]} ({row[1]})]\n{row[2]}\n{row[3]}\n\n"
                                conn.close()
                        except Exception as e:
                            print(f"  (Logic decision error: {e})")
                    file_hash = get_file_hash(f)
                    with tracker_lock:
                        if tracker.is_processed(file_hash) and verified_flag:
                            promote_verified_file(file_hash, f.name, source_file=f)
                            tracker.processed_count += 1
                            return None
                    success = process_file(f, tracker, logic_context=logic_context)
                    with tracker_lock:
                        tracker.processed_count += 1
                        if success and verified_flag:
                            file_hash = get_file_hash(f)
                            promote_verified_file(file_hash, f.name, source_file=f)
                    gc.collect()
                    return None

                if limit_files is not None:
                    files = files[:limit_files]
                if dry_run:
                    for f in files:
                        print(f"[DRY-RUN] Would process: {f.name}")
                else:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=getattr(config, 'PARALLEL_INGESTION_WORKERS', getattr(config, 'PARALLEL_WORKERS', 1))) as executor:
                        list(executor.map(process_one, files))
            else:
                file_count = 0
                for f in files:
                    if limit_files is not None and file_count >= limit_files:
                        print(f"Reached limit of {limit_files} files, stopping.")
                        break
                    file_count += 1
                    if dry_run:
                        print(f"[DRY-RUN] Would process: {f.name}")
                        continue
                    # sequential code preserved as before
                    logic_context = ""
                    if logic_mode:
                        try:
                            result = extract_text_from_file(f)
                            first_text = result["text"][:1000]
                            logic_ids = decide_logic_modules(first_text, context=first_text)
                            if logic_ids:
                                conn = db.db_connect("logic")
                                cur = conn.cursor()
                                for lid in logic_ids:
                                    cur.execute("SELECT name, category, summary, content FROM logic_modules WHERE logic_id=?", (lid,))
                                    row = cur.fetchone()
                                    if row:
                                        logic_context += f"[Logic: {row[0]} ({row[1]})]\n{row[2]}\n{row[3]}\n\n"
                                conn.close()
                        except Exception as e:
                            print(f"  (Logic decision error: {e})")
                    file_hash = get_file_hash(f)
                    if tracker.is_processed(file_hash) and verified_flag:
                        promote_verified_file(file_hash, f.name, source_file=f)
                        tracker.processed_count += 1
                        gc.collect()
                        time.sleep(0.1)
                        continue
                    success = process_file(f, tracker, logic_context=logic_context)
                    tracker.processed_count += 1
                    if success and verified_flag:
                        file_hash = get_file_hash(f)
                        promote_verified_file(file_hash, f.name, source_file=f)
                    gc.collect()
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            os._exit(0)
        print(f"\nGuided learning complete. Processed {tracker.processed_count} files.")
        return

    print("No mode specified.")
if __name__ == "__main__":
    main()