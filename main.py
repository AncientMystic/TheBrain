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
                        config.TQDM_AVAILABLE = True
        except ImportError:
            config.TQDM_AVAILABLE = False
            print("tqdm not installed; falling back to normal prints.")

from core import db
from core.progress import ProgressTracker
from core.file_utils import get_file_hash
from core.embeddings import get_embeddings_batch
from core.write_queue import enqueue_many, flush_writes
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
from logic import decide_logic_modules
from reasoning.orchestrator import orchestrate_reasoning
from deep_research.recoll_guided_learning import run_recoll_guided_learning
import threading
def clean_answer_citations(answer, reference_names):
    """Replace any [doc:n] with [n] and ensure a References section exists."""
    import re
    answer = re.sub(r'\[doc\s*:\s*(\d+)\]', r'[\1]', answer)
    if "References" not in answer and reference_names:
        answer += "\n\n### References\n"
        for i, name in enumerate(reference_names, 1):
            answer += f"{i}. {name}\n"
    return answer

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
    except Exception as e:
        logger.warning(f"Handled exception: {e}", exc_info=True)
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


def process_file(filepath, tracker, logic_context="", preloaded=None):
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
        if preloaded is not None:
            text = preloaded["text"]
            metadata = preloaded["metadata"]
            file_format = preloaded["format"]
            ocr_used = preloaded.get("ocr_used", False)
            chunks = preloaded["chunks"]
            chunk_embs = preloaded["chunk_embs"]
            print("  (Using prefetched data)")
        else:
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
                ocr_used = result.get("ocr_used", False)
                print("  (Using cached text)")
            else:
                result = extract_text_from_file(filepath)
                text = result["text"]
                metadata = result["metadata"]
                file_format = result["format"]
                ocr_used = result.get("ocr_used", False)
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
                       ocr_used=ocr_used, page_count=None)
        conn.commit()
        if preloaded is None:
            # Check if chunks already exist for this document (cache)
            conn_idx = db.db_connect("index")
            cur_idx = conn_idx.cursor()
            cur_idx.execute("SELECT chunk_text FROM document_chunks WHERE doc_hash=? ORDER BY chunk_index", (file_hash,))
            cached_chunk_rows = cur_idx.fetchall()
            conn_idx.close()
            if cached_chunk_rows:
                chunks = [r["chunk_text"] for r in cached_chunk_rows]
                print("  (Reusing cached chunks)")
            else:
                chunks = chunk_document(text)
            print(f"  Created {len(chunks)} chunks")
            store_chunks(conn, file_hash, chunks)
            conn.commit()
            conn.close()

            print("  Generating embeddings...")
            chunk_embs = get_embeddings_batch(chunks, batch_size=config.EMBEDDING_BATCH_SIZE, space='hyperbolic')
        else:
            # Preloaded path (fixed): persist chunks not yet stored, reuse embeddings.
            try:
                conn_idx = db.db_connect("index")
                cur_idx = conn_idx.cursor()
                cur_idx.execute("SELECT COUNT(*) AS n FROM document_chunks WHERE doc_hash=?", (file_hash,))
                _nrow = cur_idx.fetchone()
                _n = _nrow["n"] if _nrow else 0
                conn_idx.close()
            except Exception:
                _n = 0
            if not _n:
                print(f"  Created {len(chunks)} chunks (prefetched)")
                store_chunks(conn, file_hash, chunks)
            conn.commit()
            conn.close()
            print("  (Reusing prefetched embeddings)")
        # Compute document embedding as hyperbolic Frechet mean of chunk embeddings
        # (chunk_embs already hyperbolic; single ensure, no double exp_map)
        if chunk_embs:
            valid_embs = [np.array(emb, dtype=np.float32) for emb in chunk_embs if emb is not None]
            if valid_embs:
                from core.hyperbolic import ensure_hyperbolic, frechet_mean
                hyperbolic_points = [ensure_hyperbolic(emb, space='hyperbolic') for emb in valid_embs]
                doc_emb_hyper = frechet_mean(hyperbolic_points)
                blob = sqlite3.Binary(doc_emb_hyper.tobytes())
                conn_emb = db.db_connect("embeddings")
                conn_emb.execute("INSERT OR REPLACE INTO document_embeddings (doc_hash, embedding, model, embedding_space) VALUES (?,?,?, 'hyperbolic')",
                                 (file_hash, blob, config.EMBEDDING_MODEL))
                conn_emb.commit(); conn_emb.close()

        print("  Extracting knowledge via LLM...")
        from core.metrics import inc_counter, Timer
        inc_counter('files_processed_total')
        with Timer('extraction_duration_seconds'):
            chunk_results = extract_from_chunks(chunks, model=None, chunk_embeddings=chunk_embs, logic_context=logic_context)
        all_extracted = {"facts": [], "entities": [], "relationships": [], "people": [], "locations": [],
                         "dates": [], "events": [], "discoveries": [], "gems": []}
        fact_chunk_map = {}  # key: (fact_text, source_span) -> chunk_id
        for chunk_idx, chunk_data in enumerate(chunk_results):
            for key in all_extracted:
                if key in chunk_data:
                    all_extracted[key].extend(chunk_data[key])
            # Map facts to chunk_id
            for fact in chunk_data.get("facts", []):
                if isinstance(fact, dict) and fact.get("fact_text"):
                    span = fact.get("source_span", "")
                    fact_chunk_map[(fact["fact_text"], span)] = chunk_idx  # store chunk index; later convert to chunk_id

        print("  Validating and deduplicating...")
        import config as _cfg2
        _min_conf = float(getattr(_cfg2, "MIN_FACT_CONFIDENCE", 0.3))
        _min_prio = float(getattr(_cfg2, "MIN_PRIORITY_CONFIDENCE", 0.2))
        _raw_n = len(all_extracted["facts"])
        _kept = []
        _dropped_conf = 0
        for f in all_extracted["facts"]:
            if not isinstance(f, dict) or not f.get("fact_text", "").strip():
                _dropped_conf += 1
                continue
            try:
                _c = float(f.get("confidence", 0.0))
            except Exception:
                _c = 0.0
            _thr = _min_prio if f.get("recall_priority") else _min_conf
            if _c < _thr:
                _dropped_conf += 1
                continue
            # Preserve span-missing for verifier (text_grounding will fail gracefully);
            # never fabricate span (breaks provenance). Flag instead.
            if not f.get("source_span"):
                f["span_missing"] = True
                try:
                    f["confidence"] = _c * 0.8
                except Exception:
                    pass
            _kept.append(f)
        print(f"  (Fact triage: raw={_raw_n} kept={len(_kept)} dropped_lowconf={_dropped_conf} min_conf={_min_conf})")
        try:
            from core.metrics import inc_counter as _inc
            _inc("facts_raw_total", _raw_n)
            _inc("facts_kept_triage_total", len(_kept))
        except Exception:
            pass
        _pre_dedup = len(_kept)
        all_extracted["facts"] = deduplicate_list(_kept, key_func=lambda f: normalize_key(f.get("fact_text","") + "||" + f.get("source_span","")))
        if len(all_extracted["facts"]) < _pre_dedup:
            print(f"  (Dedup: {_pre_dedup} -> {len(all_extracted['facts'])} by text+span)")

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
        total_facts = len(all_extracted["facts"])
        fact_progress_count = 0
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
            except Exception as e:
                logger.warning(f"Handled exception: {e}", exc_info=True)
                pass

            # VeriCoT formal representation (if possible)
            try:
                from reasoning.verify import extract_triple_from_text
                triple = extract_triple_from_text(fact.get("fact_text", ""))
                if triple and all(k in triple for k in ("subject", "predicate", "object")):
                    formal_repr = json.dumps(triple)
            except Exception as e:
                logger.warning(f"Handled exception: {e}", exc_info=True)
                pass

            # R-CoT verification for lower confidence facts
            if fact.get("confidence", 0) < 0.7:
                try:
                    from reasoning.verify import verify_rcot
                    if verify_rcot(fact.get("fact_text", ""), None):
                        rcot_verified = 1
                except Exception as e:
                    logger.warning(f"Handled exception: {e}", exc_info=True)
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
            fact_progress_count += 1
            if fact_progress_count % 10 == 0 or fact_progress_count == total_facts:
                print(f"\r    Storing facts: {fact_progress_count}/{total_facts}", end="", flush=True)
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

        print()  # newline after fact storage progress
        if fact_rows:
            # Disable FTS triggers for bulk insert
            cur_facts.execute("DROP TRIGGER IF EXISTS key_facts_ai")
            cur_facts.execute("DROP TRIGGER IF EXISTS key_facts_au")
            cur_facts.execute("DROP TRIGGER IF EXISTS key_facts_ad")
            enqueue_many(
                "key_facts",
                "INSERT INTO key_facts (doc_hash, doc_name, fact_type, fact_text, canonical_value, source_span, confidence, verified, symstep_contradiction, formal_representation, rcot_verified) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                fact_rows
            )
            # Flush immediately so fact IDs are available for subsequent mapping
            flush_writes()
            # Rebuild FTS table
            cur_facts.execute("""
                INSERT INTO key_facts_fts(rowid, fact_text, canonical_value, source_span)
                SELECT fact_id, fact_text, canonical_value, source_span FROM key_facts
                WHERE fact_id NOT IN (SELECT rowid FROM key_facts_fts)
            """)
            # Recreate triggers
            cur_facts.execute("""
                CREATE TRIGGER IF NOT EXISTS key_facts_ai AFTER INSERT ON key_facts BEGIN
                    INSERT INTO key_facts_fts(rowid, fact_text, canonical_value, source_span)
                    VALUES (new.fact_id, new.fact_text, new.canonical_value, new.source_span);
                END;
            """)
            cur_facts.execute("""
                CREATE TRIGGER IF NOT EXISTS key_facts_ad AFTER DELETE ON key_facts BEGIN
                    DELETE FROM key_facts_fts WHERE rowid = old.fact_id;
                END;
            """)
            cur_facts.execute("""
                CREATE TRIGGER IF NOT EXISTS key_facts_au AFTER UPDATE ON key_facts BEGIN
                    DELETE FROM key_facts_fts WHERE rowid = old.fact_id;
                    INSERT INTO key_facts_fts(rowid, fact_text, canonical_value, source_span)
                    VALUES (new.fact_id, new.fact_text, new.canonical_value, new.source_span);
                END;
            """)

        # Retrieve last inserted IDs
        cur_facts.execute("SELECT fact_id, source_span, canonical_value FROM key_facts WHERE doc_hash=? ORDER BY fact_id DESC LIMIT ?",
                          (file_hash, len(fact_rows)))
        inserted = cur_facts.fetchall()[::-1]

        # Pre-fetch chunk_id_by_index map for this document
        chunk_id_by_index = {}
        cur_index.execute("SELECT chunk_id, chunk_index FROM document_chunks WHERE doc_hash=?", (file_hash,))
        for cid, cidx in cur_index.fetchall():
            chunk_id_by_index[cidx] = cid

        for fact, inserted_row in zip(all_extracted["facts"], inserted):
            fact_id = inserted_row["fact_id"]
            span = fact.get("source_span","")
            chunk_key = (fact.get("fact_text",""), span)
            cidx = fact_chunk_map.get(chunk_key)
            chunk_id = chunk_id_by_index.get(cidx) if cidx is not None else None
            if chunk_id:
                source_rows.append((fact_id, file_hash, chunk_id, span, span))
            if fact.get("canonical_value"):
                try:
                    from core.fact_normalizer import normalize_name
                    norm = normalize_name(fact["canonical_value"])
                    entity_index_rows.append((fact_id, fact["canonical_value"], norm))
                except ImportError:
                    logger.warning(f"Handled exception: {e}", exc_info=True)
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
        # Store all categories with executemany
        entity_rows = [(_safe_str_for_db(e.get("entity_type", "OTHER")), _safe_str_for_db(e.get("entity_name", "")), _safe_str_for_db(e.get("normalized_name", "")), _safe_str_for_db(e.get("source_span", "")), e.get("confidence", 0.0)) for e in all_extracted.get("entities", [])]
        if entity_rows:
            conn_facts.executemany("INSERT INTO entities (doc_hash, entity_type, entity_name, normalized_name, source_span, confidence) VALUES (?,?,?,?,?,?)",
                                   [(file_hash, *row) for row in entity_rows])

        person_rows = [(_safe_str_for_db(p.get("person_name", "")), _safe_str_for_db(p.get("normalized_name", "")), _safe_str_for_db(p.get("role", "")), _safe_str_for_db(p.get("source_span", "")), p.get("confidence", 0.0)) for p in all_extracted.get("people", [])]
        if person_rows:
            conn_facts.executemany("INSERT INTO people (doc_hash, person_name, normalized_name, role, source_span, confidence) VALUES (?,?,?,?,?,?)",
                                   [(file_hash, *row) for row in person_rows])

        location_rows = [(_safe_str_for_db(l.get("location_name", "")), _safe_str_for_db(l.get("normalized_place", "")), _safe_str_for_db(l.get("location_type", "")), _safe_str_for_db(l.get("source_span", "")), l.get("confidence", 0.0)) for l in all_extracted.get("locations", [])]
        if location_rows:
            conn_facts.executemany("INSERT INTO locations (doc_hash, location_name, normalized_place, location_type, source_span, confidence) VALUES (?,?,?,?,?,?)",
                                   [(file_hash, *row) for row in location_rows])

        date_rows = [(_safe_str_for_db(d.get("date_text", "")), _safe_str_for_db(d.get("normalized_date", "")), _safe_str_for_db(d.get("date_type", "")), _safe_str_for_db(d.get("source_span", "")), d.get("confidence", 0.0)) for d in all_extracted.get("dates", [])]
        if date_rows:
            conn_facts.executemany("INSERT INTO dates (doc_hash, date_text, normalized_date, date_type, source_span, confidence) VALUES (?,?,?,?,?,?)",
                                   [(file_hash, *row) for row in date_rows])

        event_rows = [(_safe_str_for_db(ev.get("event_name", "")), _safe_str_for_db(ev.get("normalized_name", "")), _safe_str_for_db(ev.get("event_date", "")), _safe_str_for_db(ev.get("event_type", "")), _safe_str_for_db(ev.get("description", "")), _safe_str_for_db(ev.get("significance", "")), _safe_str_for_db(ev.get("source_span", "")), ev.get("confidence", 0.0)) for ev in all_extracted.get("events", [])]
        if event_rows:
            conn_facts.executemany("INSERT INTO events (doc_hash, event_name, normalized_name, event_date, event_type, description, significance, source_span, confidence) VALUES (?,?,?,?,?,?,?,?,?)",
                                   [(file_hash, *row) for row in event_rows])

        discovery_rows = [(_safe_str_for_db(d.get("discovery_name", "")), _safe_str_for_db(d.get("normalized_name", "")), _safe_str_for_db(d.get("description", "")), _safe_str_for_db(d.get("date", "")), _safe_str_for_db(d.get("significance", "")), _safe_str_for_db(d.get("source_span", "")), d.get("confidence", 0.0)) for d in all_extracted.get("discoveries", [])]
        if discovery_rows:
            conn_facts.executemany("INSERT INTO discoveries (doc_hash, discovery_name, normalized_name, description, date, significance, source_span, confidence) VALUES (?,?,?,?,?,?,?,?)",
                                   [(file_hash, *row) for row in discovery_rows])

        gem_rows = [(_safe_str_for_db(g.get("gem_text", "")), _safe_str_for_db(g.get("category", "")), g.get("importance", 0.0), _safe_str_for_db(g.get("source_span", "")), g.get("confidence", 0.0)) for g in all_extracted.get("gems", [])]
        if gem_rows:
            conn_facts.executemany("INSERT INTO gems (doc_hash, gem_text, category, importance, source_span, confidence) VALUES (?,?,?,?,?,?)",
                                   [(file_hash, *row) for row in gem_rows])
        # Flush any remaining queued writes
        flush_writes()
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
        # Release pooled connections to keep memory flat
        try:
            from core import db as db_module
            db_module.close_all_connections()
        except Exception as e:
            logger.warning(f"Handled exception: {e}", exc_info=True)
            pass
        return True
    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")
        traceback.print_exc()
        # Close all DB connections before marking error to avoid lock
        try:
            from core import db as db_module
            db_module.close_all_connections()
        except Exception as e:
            logger.warning(f"Handled exception: {e}", exc_info=True)
            pass
        tracker.mark_error(file_hash, stage="processing")
        logger.warning("Unexpected exception occurred", exc_info=True)
        return False


# Helper functions for storing extracted categories (moved here to avoid circular import)
def _store_entity(conn, doc_hash, file_name, entity):
    enqueue_write("key_facts",
                  "INSERT INTO entities (doc_hash, entity_type, entity_name, normalized_name, source_span, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                  (doc_hash, _safe_str_for_db(entity.get("entity_type", "OTHER")), _safe_str_for_db(entity.get("entity_name", "")),
                   _safe_str_for_db(entity.get("normalized_name", "")), _safe_str_for_db(entity.get("source_span", "")), entity.get("confidence", 0.0)))

def _store_person(conn, doc_hash, file_name, person):
    enqueue_write("key_facts",
                  "INSERT INTO people (doc_hash, person_name, normalized_name, role, source_span, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                  (doc_hash, _safe_str_for_db(person.get("person_name", "")), _safe_str_for_db(person.get("normalized_name", "")),
                   _safe_str_for_db(person.get("role", "")), _safe_str_for_db(person.get("source_span", "")), person.get("confidence", 0.0)))

def _store_location(conn, doc_hash, file_name, location):
    enqueue_write("key_facts",
                  "INSERT INTO locations (doc_hash, location_name, normalized_place, location_type, source_span, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                  (doc_hash, _safe_str_for_db(location.get("location_name", "")), _safe_str_for_db(location.get("normalized_place", "")),
                   _safe_str_for_db(location.get("location_type", "")), _safe_str_for_db(location.get("source_span", "")), location.get("confidence", 0.0)))

def _store_date(conn, doc_hash, file_name, date):
    enqueue_write("key_facts",
                  "INSERT INTO dates (doc_hash, date_text, normalized_date, date_type, source_span, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                  (doc_hash, _safe_str_for_db(date.get("date_text", "")), _safe_str_for_db(date.get("normalized_date", "")),
                   _safe_str_for_db(date.get("date_type", "")), _safe_str_for_db(date.get("source_span", "")), date.get("confidence", 0.0)))

def _store_event(conn, doc_hash, file_name, event):
    enqueue_write("key_facts",
                  "INSERT INTO events (doc_hash, event_name, normalized_name, event_date, event_type, description, significance, source_span, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (doc_hash, _safe_str_for_db(event.get("event_name", "")), _safe_str_for_db(event.get("normalized_name", "")),
                   _safe_str_for_db(event.get("event_date", "")), _safe_str_for_db(event.get("event_type", "")), _safe_str_for_db(event.get("description", "")),
                   _safe_str_for_db(event.get("significance", "")), _safe_str_for_db(event.get("source_span", "")), event.get("confidence", 0.0)))

def _store_discovery(conn, doc_hash, file_name, discovery):
    enqueue_write("key_facts",
                  "INSERT INTO discoveries (doc_hash, discovery_name, normalized_name, description, date, significance, source_span, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (doc_hash, _safe_str_for_db(discovery.get("discovery_name", "")), _safe_str_for_db(discovery.get("normalized_name", "")),
                   _safe_str_for_db(discovery.get("description", "")), _safe_str_for_db(discovery.get("date", "")), _safe_str_for_db(discovery.get("significance", "")),
                   _safe_str_for_db(discovery.get("source_span", "")), discovery.get("confidence", 0.0)))

def _store_gem(conn, doc_hash, file_name, gem):
    enqueue_write("key_facts",
                  "INSERT INTO gems (doc_hash, gem_text, category, importance, source_span, confidence) VALUES (?, ?, ?, ?, ?, ?)",
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
        details = row["details"] or row["resolved_by"] or "No details available"
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
            except Exception as e:
                logger.warning(f"Handled exception: {e}", exc_info=True)
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








# Precompiled regexes for direct_document_lookup
import re as _re
import logging
logger = logging.getLogger(__name__)
_EPISODE_PATTERN = _re.compile(r'(?:episode|ep|#)\s*(\d{2,3})', _re.IGNORECASE)
_LIST_PATTERN = _re.compile(r'(?:list|what|which).*?episodes', _re.IGNORECASE)
_SEARCH_TERM_PATTERN = _re.compile(r'(?:for|of|about)\s*(.+?)\s*$', _re.IGNORECASE)
_FALLBACK_NUM_PATTERN = _re.compile(r'\b(\d{2,3})\b')

def direct_document_lookup(query, top_k=1000):
    """Retrieve documents based on direct references in query (episode numbers, list requests)."""
    import re
    from core import db

    doc_hashes = []
    ep_match = _EPISODE_PATTERN.search(query)
    list_match = _LIST_PATTERN.search(query)

    if ep_match:
        num = ep_match.group(1)
        conn = db.db_connect("index")
        cur = conn.cursor()
        cur.execute("SELECT file_hash, filename, title FROM documents WHERE filename LIKE ? OR title LIKE ?",
                    (f'%{num}%', f'%{num}%'))
        rows = cur.fetchall()
        conn.close()
        doc_hashes.extend([r['file_hash'] for r in rows])
    elif list_match:
        # Extract a broader search term (default to "why files" if none)
        search_term = ''
        m = _SEARCH_TERM_PATTERN.search(query)
        if m:
            search_term = m.group(1).strip().lower().rstrip('?!.')
        if not search_term:
            search_term = "why files"
        # Normalize: remove leading "the " if present
        if search_term.startswith("the "):
            search_term = search_term[4:]
        conn = db.db_connect("index")
        cur = conn.cursor()
        cur.execute("SELECT file_hash, filename, title FROM documents WHERE filename LIKE ? OR title LIKE ? ORDER BY filename",
                    (f'%{search_term}%', f'%{search_term}%'))
        rows = cur.fetchall()
        conn.close()
        doc_hashes.extend([r['file_hash'] for r in rows])
    else:
        # Fallback numeric
        num_match = _FALLBACK_NUM_PATTERN.search(query)
        if num_match:
            num = num_match.group(1)
            conn = db.db_connect("index")
            cur = conn.cursor()
            cur.execute("SELECT file_hash, filename, title FROM documents WHERE filename LIKE ? OR title LIKE ?",
                        (f'%{num}%', f'%{num}%'))
            rows = cur.fetchall()
            conn.close()
            doc_hashes.extend([r['file_hash'] for r in rows])

    if not doc_hashes:
        return [], []

    facts = []
    chunks = []
    seen_facts = set()
    seen_chunks = set()

    # If list request, produce synthetic facts listing all documents
    if list_match:
        for dh in doc_hashes:
            conn = db.db_connect("index")
            cur = conn.cursor()
            cur.execute("SELECT title, filename FROM documents WHERE file_hash=?", (dh,))
            row = cur.fetchone()
            conn.close()
            if row:
                display = row['title'] or row['filename']
                facts.append({
                    'fact_id': f"doc_{dh}",
                    'doc_hash': dh,
                    'doc_name': display,
                    'fact_text': f"Available episode/document: {display}",
                    'canonical_value': display,
                    'source_span': '',
                    'confidence': 1.0,
                })
        return facts, chunks

    # Otherwise fetch facts/chunks for matched documents
    for dh in doc_hashes:
        conn = db.db_connect("key_facts")
        cur = conn.cursor()
        cur.execute("SELECT fact_id, doc_hash, doc_name, fact_text, canonical_value, source_span, confidence FROM key_facts WHERE doc_hash=? ORDER BY confidence DESC LIMIT 200", (dh,))
        frows = cur.fetchall()
        conn.close()
        for f in frows:
            fid = f['fact_id']
            if fid not in seen_facts:
                seen_facts.add(fid)
                facts.append({
                    'fact_id': fid,
                    'doc_hash': f['doc_hash'],
                    'doc_name': f['doc_name'],
                    'fact_text': f['fact_text'],
                    'canonical_value': f['canonical_value'],
                    'source_span': f['source_span'],
                    'confidence': f['confidence'],
                })
        conn = db.db_connect("index")
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, doc_hash, chunk_text FROM document_chunks WHERE doc_hash=? ORDER BY chunk_index LIMIT 30", (dh,))
        crows = cur.fetchall()
        conn.close()
        for c in crows:
            cid = c['chunk_id']
            if cid not in seen_chunks:
                seen_chunks.add(cid)
                chunks.append((0, cid, c['doc_hash'], c['chunk_text']))
    return facts, chunks



def _load_or_extract_text(file_hash, filepath):
    """Return (text, metadata, ocr_used), using document_text_cache when available."""
    conn = db.db_connect("index")
    try:
        cur = conn.cursor()
        cur.execute("SELECT text FROM document_text_cache WHERE file_hash=?", (file_hash,))
        row = cur.fetchone()
    finally:
        conn.close()
    if row:
        return row["text"], {}, False
    result = extract_text_from_file(filepath)
    text = result.get("text", "")
    metadata = result.get("metadata", {}) or {}
    ocr_used = bool(result.get("ocr_used", False))
    if text:
        conn = db.db_connect("index")
        try:
            conn.execute("INSERT OR REPLACE INTO document_text_cache (file_hash, text, metadata_json) VALUES (?,?,?)",
                         (file_hash, text, json.dumps(metadata)))
            conn.commit()
        finally:
            conn.close()
    return text, metadata, ocr_used


def _load_or_chunk(file_hash, text):
    """Return chunks, using cached document_chunks when available."""
    conn = db.db_connect("index")
    try:
        cur = conn.cursor()
        cur.execute("SELECT chunk_text FROM document_chunks WHERE doc_hash=?", (file_hash,))
        cached_chunks = cur.fetchall()
    finally:
        conn.close()
    if cached_chunks:
        return [r["chunk_text"] for r in cached_chunks]
    return chunk_document(text)


def prepare_next_file(filepath):
    """Extract text, chunk, and embed for a file (CPU/IO bound). Single canonical implementation."""
    file_hash = get_file_hash(filepath)
    try:
        text, metadata, ocr_used = _load_or_extract_text(file_hash, filepath)
        if not text:
            return None
        chunks = _load_or_chunk(file_hash, text)
        # Embeddings — always hyperbolic per design; batch size from config, not hardcoded
        chunk_embs = get_embeddings_batch(chunks, batch_size=config.EMBEDDING_BATCH_SIZE, space='hyperbolic')
        return {
            "file": filepath,
            "file_hash": file_hash,
            "text": text,
            "chunks": chunks,
            "chunk_embs": chunk_embs,
            "metadata": metadata,
            "format": Path(filepath).suffix.lstrip(".").lower(),
            "ocr_used": ocr_used,
        }
    except Exception as e:
        fname = getattr(filepath, "name", str(filepath))
        print(f"    (Prefetch error for {fname}: {e})")
        logger.warning("Unexpected exception occurred", exc_info=True)
        return None

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
    retry_failed = "--retry-failed" in sys.argv
    build_recoll_index = "--build-recoll-index" in sys.argv
    train_gnn_flag = "--train-gnn" in sys.argv
    maintenance_mode = "--maintenance" in sys.argv
    interactive = "--interactive" in sys.argv
    session_id = None
    if "--session" in sys.argv:
        idx = sys.argv.index("--session") + 1
        if idx < len(sys.argv):
            session_id = sys.argv[idx]
    input_path = None
    if "--input" in sys.argv:
        idx = sys.argv.index("--input") + 1
        if idx < len(sys.argv):
            raw_input = sys.argv[idx]
            try:
                # Resolve to absolute to avoid traversal surprises; generic, not doc-specific.
                # Allow any existing path by default, but log resolved value for audit.
                # Strict allow-list only when THEBRAIN_ALLOWED_ROOTS env is set (comma-separated).
                from pathlib import Path as _Path
                import os as _os
                _resolved = _Path(raw_input).expanduser().resolve()
                _allowed = [r.strip() for r in _os.environ.get("THEBRAIN_ALLOWED_ROOTS", "").split(",") if r.strip()]
                if _allowed:
                    _ar = [str(_Path(r).expanduser().resolve()) for r in _allowed]
                    if not any(str(_resolved).startswith(a) for a in _ar) and "--allow-outside-root" not in sys.argv:
                        print(f"[ERROR] --input {raw_input} outside THEBRAIN_ALLOWED_ROOTS. Use --allow-outside-root to override.")
                        sys.exit(2)
                input_path = str(_resolved)
            except SystemExit:
                raise
            except Exception as e:
                print(f"[ERROR] Invalid --input path: {e}")
                sys.exit(2)

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
    # removed duplicate assignment below
    if "--recoll-query" in sys.argv:
        idx = sys.argv.index("--recoll-query") + 1
        if idx < len(sys.argv):
            import os as _os2
            _maxq = int(_os2.environ.get("RECOLL_MAX_QUERY_CHARS", "500"))
            recoll_query = str(sys.argv[idx]).strip()[:_maxq]

    recoll_limit = None
    if "--recoll-limit" in sys.argv:
        idx = sys.argv.index("--recoll-limit") + 1
        if idx < len(sys.argv):
            try:
                recoll_limit = max(1, min(int(sys.argv[idx]), 200))
            except ValueError:
                print("Invalid --recoll-limit value, using default.")

    preview_chars = None
    if "--preview-chars" in sys.argv:
        idx = sys.argv.index("--preview-chars") + 1
        if idx < len(sys.argv):
            try:
                preview_chars = max(100, min(int(sys.argv[idx]), 10000))
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
    # Preload FastExtractor once before any processing (if guided and enabled)
    if guided and config.FAST_EXTRACTOR_ENABLED:
        try:
            import extraction.llm_extractor as lle
            from fast_extractor.hybrid_extractor import FastExtractor
            lle._fast_extractor_instance = FastExtractor()
            print("  (Preloaded ONNX FastExtractor)")
        except Exception as e:
            print(f"  (FastExtractor preload error: {e})")

    print("Initialization complete. Preparing processing...")

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
                    logger.warning(f"Handled exception: {e}", exc_info=True)
                    pass

        train_gates = "--train-gates" in sys.argv
        train_distilled = "--train-distilled" in sys.argv
        once = "--once" in sys.argv

        if once:
            print("Running maintenance once...")
            run_maintenance_once(train_gates=train_gates, train_distilled=train_distilled)
            print("Maintenance finished.")
        else:
            start_background_maintenance(interval_seconds=interval, train_gates=train_gates, train_distilled=train_distilled)
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
        from chat.conversation import add_message, get_hyperbolic_conversation_history
        from deep_research.coordinator import DeepResearchCoordinator

        print("Chat mode. Type 'exit' to quit. Add --deep-research to enable autonomous research.")
        if session_id is None:
            session_id = f"cli_{int(time.time())}"
        else:
            print(f"Resuming session: {session_id}")
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
                conversation_history = get_hyperbolic_conversation_history(session_id, query)

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
                                logic_context += f"[Logic: {row[0]} ({row[1]})\n{row[2]}\n{row[3]}\n\n"
                        conn.close()

                    memories = retrieve_memories(query, top_k=5, session_id=session_id)
                    memory_text = "\n".join([f"[Memory] {m[2]}" for m in memories])

                    analysis = analyze_query(query)

                    # Direct document lookup for episode/numeric references
                    direct_facts, direct_chunks = direct_document_lookup(query)

                    facts = retrieve_from_graph(analysis, top_k=150, max_depth=2)
                    chunks = fallback_to_chunks(query, top_k=12)

                    # Merge direct facts
                    existing_fact_ids = {f.get('fact_id') for f in facts if f.get('fact_id')}
                    for df in direct_facts:
                        if df.get('fact_id') not in existing_fact_ids:
                            facts.append(df)
                            existing_fact_ids.add(df.get('fact_id'))

                    # Merge direct chunks
                    existing_chunk_ids = {c[0] for c in chunks if c[0]}
                    for dc in direct_chunks:
                        if dc[0] not in existing_chunk_ids:
                            chunks.append(dc)
                            existing_chunk_ids.add(dc[0])

                    context = build_context(facts, chunks=chunks, conversation_history=conversation_history)
                    if logic_context:
                        context = logic_context + "\n\n" + context
                    if memory_text:
                        context = memory_text + "\n\n" + context

                    answer = generate_answer(query, context)

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

    if retry_failed:
        from core.progress import get_pending_retry_files
        pending = get_pending_retry_files()
        if not pending:
            print("No pending retry files.")
            return
        print(f"Found {len(pending)} files to retry.")
        tracker = ProgressTracker()
        for file_path in pending:
            path = Path(file_path)
            if not path.exists():
                print(f"File missing: {path}")
                continue
            print(f"Retrying: {path}")
            # Reset status to pending so is_processed returns False
            file_hash = get_file_hash(path)
            tracker.mark_processed(file_hash, status="pending", stage="retry")
            success = process_file(path, tracker)
            if success:
                tracker.mark_processed(file_hash)
        print("Retry complete.")
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
        print(f"Found {len(files)} files to process.")
        # Curriculum ordering
        if getattr(config, "USE_HYPERBOLIC_CURRICULUM", True):
            print("Ordering files by hyperbolic curriculum...")
            try:
                from learning.curriculum import order_files_by_curriculum
                files = order_files_by_curriculum(files)
                print(f"  (Ordered {len(files)} files by hyperbolic curriculum)")
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"    (Curriculum ordering error: {e})")
        total = len(files)
        # Preload FastExtractor once before document loop
        if config.FAST_EXTRACTOR_ENABLED:
            try:
                import extraction.llm_extractor as lle
                from fast_extractor.hybrid_extractor import FastExtractor
                lle._fast_extractor_instance = FastExtractor()
                print("  (Preloaded ONNX FastExtractor)")
            except Exception as e:
                print(f"  (FastExtractor preload error: {e})")

        print("Starting document processing...")
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
                    # Single extraction per file (no double parse/OCR): prepare once,
                    # reuse text for logic decision + pass preloaded into process_file.
                    try:
                        _prep = prepare_next_file(f)
                    except Exception as e:
                        print(f"  (Prepare error for {getattr(f, 'name', f)}: {e})")
                        _prep = None
                    logic_context = ""
                    if logic_mode and _prep and _prep.get("text"):
                        try:
                            first_text = _prep["text"][:1000]
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
                    success = process_file(f, tracker, logic_context=logic_context, preloaded=_prep)
                    with tracker_lock:
                        tracker.processed_count += 1
                        if success and verified_flag:
                            file_hash = get_file_hash(f)
                            promote_verified_file(file_hash, f.name, source_file=f)
                        # Smart GC: every N files only (no per-file stall, no sleep)
                        try:
                            _every = int(getattr(config, "GC_EVERY_N_FILES", 25))
                            if tracker.processed_count % max(1, _every) == 0:
                                gc.collect()
                        except Exception:
                            pass
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
                prefetched_data = None
                prefetch_lock = threading.Lock()
                for idx, f in enumerate(files):
                    if limit_files is not None and file_count >= limit_files:
                        print(f"Reached limit of {limit_files} files, stopping.")
                        break
                    file_count += 1
                    if dry_run:
                        print(f"[DRY-RUN] Would process: {f.name}")
                        continue
                    # Use prefetched data if available (resolved-path identity, not object identity)
                    preloaded = None
                    try:
                        from pathlib import Path as _P
                        _cur_res = str(_P(f).expanduser().resolve())
                    except Exception:
                        _cur_res = str(f)
                    if getattr(config, "PREFETCH_NEXT_DOCUMENT", False) and prefetched_data:
                        try:
                            from pathlib import Path as _P2
                            _pre_res = str(_P2(prefetched_data.get("file", "")).expanduser().resolve())
                        except Exception:
                            _pre_res = str(prefetched_data.get("file", ""))
                        if _pre_res == _cur_res:
                            preloaded = prefetched_data
                            prefetched_data = None
                            print("  (Using prefetched data)")
                    # sequential logic reuses prefetched/preloaded text (no double parse/OCR)
                    logic_context = ""
                    if logic_mode:
                        try:
                            _ltext = None
                            if preloaded and preloaded.get("text"):
                                _ltext = preloaded["text"][:1000]
                            else:
                                _lr = extract_text_from_file(f)
                                _ltext = _lr["text"][:1000]
                            logic_ids = decide_logic_modules(_ltext, context=_ltext)
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
                        try:
                            _every0 = int(getattr(config, "GC_EVERY_N_FILES", 25))
                            if tracker.processed_count % max(1, _every0) == 0:
                                gc.collect()
                        except Exception:
                            pass
                        continue
                    # Spawn prefetch for next file
                    if getattr(config, "PREFETCH_NEXT_DOCUMENT", False) and idx + 1 < len(files):
                        next_file = files[idx + 1]
                        def do_prefetch():
                            nonlocal prefetched_data
                            with prefetch_lock:
                                if prefetched_data is None:
                                    prefetched_data = prepare_next_file(next_file)
                        t = threading.Thread(target=do_prefetch, daemon=True)
                        t.start()
                    success = process_file(f, tracker, logic_context=logic_context, preloaded=preloaded)
                    tracker.processed_count += 1
                    if success and verified_flag:
                        file_hash = get_file_hash(f)
                        promote_verified_file(file_hash, f.name, source_file=f)
                    try:
                        _every1 = int(getattr(config, "GC_EVERY_N_FILES", 25))
                        if tracker.processed_count % max(1, _every1) == 0:
                            gc.collect()
                    except Exception:
                        pass
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            os._exit(0)
        print(f"\nGuided learning complete. Processed {tracker.processed_count} files.")
        return

    print("No mode specified.")
if __name__ == "__main__":
    main()
