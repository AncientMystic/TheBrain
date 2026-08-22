import os
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
import sys, time, gc, json, traceback, sqlite3, re, os
from pathlib import Path
import numpy as np
import config

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
    return re.sub(r'\s+', ' ', text.lower()).strip()


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
    file_hash = get_file_hash(filepath)
    if tracker.is_processed(file_hash):
        print(f"Skipping already processed: {filepath.name}")
        return False
    print(f"\n[{tracker.processed_count}/{tracker.total_files}] Processing: {filepath}")
    try:
        result = extract_text_from_file(filepath)
        text = result["text"]
        metadata = result["metadata"]
        file_format = result["format"]
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
        doc_emb = get_embeddings_batch([text])[0]
        if doc_emb:
            blob = sqlite3.Binary(np.array(doc_emb, dtype=np.float32).tobytes())
            conn_emb = db.db_connect("embeddings")
            conn_emb.execute("INSERT OR REPLACE INTO document_embeddings (doc_hash, embedding, model) VALUES (?,?,?)",
                             (file_hash, blob, config.EMBEDDING_MODEL))
            conn_emb.commit(); conn_emb.close()
        chunk_embs = get_embeddings_batch(chunks, batch_size=config.EMBEDDING_BATCH_SIZE)
        conn_emb = db.db_connect("embeddings")
        cur_emb = conn_emb.cursor()
        conn_index = db.db_connect("index")
        cur_index = conn_index.cursor()
        for i, (chunk_text, emb) in enumerate(zip(chunks, chunk_embs)):
            if emb:
                cur_index.execute("SELECT chunk_id FROM document_chunks WHERE doc_hash=? AND chunk_index=?",
                                  (file_hash, i))
                row = cur_index.fetchone()
                if row:
                    chunk_id = row[0]
                    blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes())
                    cur_emb.execute("INSERT OR REPLACE INTO chunk_embeddings (chunk_id, doc_hash, chunk_text, embedding, model) VALUES (?,?,?,?,?)",
                                    (chunk_id, file_hash, chunk_text, blob, config.EMBEDDING_MODEL))
        conn_emb.commit(); conn_emb.close(); conn_index.close()

        print("  Extracting knowledge via LLM...")
        chunk_results = extract_from_chunks(chunks, model=None, chunk_embeddings=chunk_embs, logic_context=logic_context)
        all_extracted = {"facts": [], "entities": [], "relationships": [], "people": [], "locations": [],
                         "dates": [], "events": [], "discoveries": [], "gems": []}
        for chunk_data in chunk_results:
            for key in all_extracted:
                if key in chunk_data:
                    all_extracted[key].extend(chunk_data[key])

        print("  Validating and deduplicating...")
        validated_facts = [f for f in all_extracted["facts"] if f.get("source_span") and f.get("source_span") in text and f.get("confidence",0) >= 0.5]
        all_extracted["facts"] = deduplicate_list(validated_facts, key_func=lambda f: normalize_key(f.get("fact_text","")))
        all_extracted["entities"] = deduplicate_list(all_extracted["entities"], key_func=lambda e: normalize_key(e.get("entity_name","")))
        all_extracted["people"] = deduplicate_list(all_extracted["people"], key_func=lambda p: normalize_key(p.get("person_name","")))
        all_extracted["locations"] = deduplicate_list(all_extracted["locations"], key_func=lambda l: normalize_key(l.get("location_name","")))
        all_extracted["dates"] = deduplicate_list(all_extracted["dates"], key_func=lambda d: normalize_key(d.get("date_text","")))
        all_extracted["events"] = deduplicate_list(all_extracted["events"], key_func=lambda e: normalize_key(e.get("event_name","")))
        all_extracted["discoveries"] = deduplicate_list(all_extracted["discoveries"], key_func=lambda d: normalize_key(d.get("discovery_name","")))
        all_extracted["gems"] = deduplicate_list(all_extracted["gems"], key_func=lambda g: normalize_key(g.get("gem_text","")))

        print("  Storing key facts...")
        conn_facts = db.db_connect("key_facts")
        cur_facts = conn_facts.cursor()
        conn_index = db.db_connect("index")
        cur_index = conn_index.cursor()
        for fact in all_extracted["facts"]:
            cur_facts.execute("INSERT INTO key_facts (doc_hash, doc_name, fact_type, fact_text, canonical_value, source_span, confidence, verified) VALUES (?,?,?,?,?,?,?,0)",
                              (file_hash, filepath.name, fact.get("fact_type"), fact.get("fact_text"), fact.get("canonical_value"), fact.get("source_span"), fact.get("confidence",0.0)))
            fact_id = cur_facts.lastrowid
            span = fact.get("source_span","")
            cur_index.execute("SELECT chunk_id FROM document_chunks WHERE doc_hash=? AND chunk_text LIKE ? LIMIT 1", (file_hash, f"%{span}%"))
            row = cur_index.fetchone()
            chunk_id = row[0] if row else None
            if chunk_id:
                cur_facts.execute("INSERT INTO fact_sources (fact_id, doc_hash, chunk_id, evidence_span, exact_quote) VALUES (?,?,?,?,?)",
                                  (fact_id, file_hash, chunk_id, span, span))
        # Store all categories
        from main import _store_entity, _store_person, _store_location, _store_date, _store_event, _store_discovery, _store_gem
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

        print("  Building hypergraph...")
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
        print(f"  Done processing {filepath.name}")
        return True
    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")
        traceback.print_exc()
        tracker.mark_error(file_hash, stage="processing")
        return False


# Helper functions for storing extracted categories (moved here to avoid circular import)
def _store_entity(conn, doc_hash, file_name, entity):
    conn.execute("""INSERT INTO entities (doc_hash, entity_type, entity_name, normalized_name, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, entity.get("entity_type", "OTHER"), entity.get("entity_name", ""),
                  entity.get("normalized_name", ""), entity.get("source_span", ""), entity.get("confidence", 0.0)))

def _store_person(conn, doc_hash, file_name, person):
    conn.execute("""INSERT INTO people (doc_hash, person_name, normalized_name, role, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, person.get("person_name", ""), person.get("normalized_name", ""),
                  person.get("role", ""), person.get("source_span", ""), person.get("confidence", 0.0)))

def _store_location(conn, doc_hash, file_name, location):
    conn.execute("""INSERT INTO locations (doc_hash, location_name, normalized_place, location_type, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, location.get("location_name", ""), location.get("normalized_place", ""),
                  location.get("location_type", ""), location.get("source_span", ""), location.get("confidence", 0.0)))

def _store_date(conn, doc_hash, file_name, date):
    conn.execute("""INSERT INTO dates (doc_hash, date_text, normalized_date, date_type, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, date.get("date_text", ""), date.get("normalized_date", ""),
                  date.get("date_type", ""), date.get("source_span", ""), date.get("confidence", 0.0)))

def _store_event(conn, doc_hash, file_name, event):
    conn.execute("""INSERT INTO events (doc_hash, event_name, normalized_name, event_date, event_type,
                                        description, significance, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                 (doc_hash, event.get("event_name", ""), event.get("normalized_name", ""),
                  event.get("event_date", ""), event.get("event_type", ""), event.get("description", ""),
                  event.get("significance", ""), event.get("source_span", ""), event.get("confidence", 0.0)))

def _store_discovery(conn, doc_hash, file_name, discovery):
    conn.execute("""INSERT INTO discoveries (doc_hash, discovery_name, normalized_name, description,
                                             date, significance, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                 (doc_hash, discovery.get("discovery_name", ""), discovery.get("normalized_name", ""),
                  discovery.get("description", ""), discovery.get("date", ""), discovery.get("significance", ""),
                  discovery.get("source_span", ""), discovery.get("confidence", 0.0)))

def _store_gem(conn, doc_hash, file_name, gem):
    conn.execute("""INSERT INTO gems (doc_hash, gem_text, category, importance, source_span, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                 (doc_hash, gem.get("gem_text", ""), gem.get("category", ""),
                  gem.get("importance", 0.0), gem.get("source_span", ""), gem.get("confidence", 0.0)))


def main():
    if "--debug" in sys.argv:
        config.DEBUG_VERBOSE = True

    server_mode = "--server" in sys.argv
    guided = "--guided-learning" in sys.argv
    chat_mode = "--chat" in sys.argv
    audit_mode = "--audit" in sys.argv
    logic_mode = "--logic" in sys.argv
    reasoning_mode = "--reasoning" in sys.argv
    deep_research = "--deep-research" in sys.argv
    recoll_mode = "--recoll" in sys.argv
    build_recoll_index = "--build-recoll-index" in sys.argv
    input_path = None
    if "--input" in sys.argv:
        idx = sys.argv.index("--input") + 1
        if idx < len(sys.argv):
            input_path = sys.argv[idx]

    validate_config()
    recoll_fast = "--recoll-fast" in sys.argv
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

    init_all()

    if recoll_fast:
        interactive = "--interactive" in sys.argv
        if recoll_query:
            # Process a single query automatically
            try:
                process_recoll_fast(recoll_query, max_results=recoll_limit, preview_chars=preview_chars)
            except ImportError:
                print("recoll_fast module not found. Please run the setup script.")
            return
        else:
            if interactive:
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
                    recoll_query = q
                    try:
                        process_recoll_fast(recoll_query, max_results=recoll_limit, preview_chars=preview_chars)
                    except ImportError:
                        print("recoll_fast module not found. Please run the setup script.")
            else:
                # Automatic mode: collect seed keywords from existing knowledge
                print("Recoll Fast automatic mode - collecting seed keywords...")
                try:
                    seeds = collect_seed_keywords(limit=config.RECOLL_AUTO_KEYWORD_LIMIT if hasattr(config, 'RECOLL_AUTO_KEYWORD_LIMIT') else 20)
                    if not seeds:
                        print("No seed keywords found in key_facts. Use --recoll-query or build knowledge first.")
                        return
                    print(f"Processing {len(seeds)} keywords automatically...")
                    for kw in seeds:
                        print(f"\n=== Processing keyword: {kw} ===")
                        process_recoll_fast(kw, max_results=recoll_limit, preview_chars=preview_chars)
                except ImportError:
                    print("recoll_fast module not found. Please run the setup script.")
            return

    if recoll_query:
        try:
            process_recoll_fast(recoll_query, max_results=recoll_limit, preview_chars=preview_chars)
        except ImportError:
            print("recoll_fast module not found. Please run the setup script.")
        return
    else:
        # Automatic mode: collect seed keywords from existing knowledge
        print("Recoll Fast automatic mode - collecting seed keywords...")
        try:
            seeds = collect_seed_keywords(limit=config.RECOLL_AUTO_KEYWORD_LIMIT if hasattr(config, 'RECOLL_AUTO_KEYWORD_LIMIT') else 20)
            if not seeds:
                print("No seed keywords found in key_facts. Use --recoll-query or build knowledge first.")
                return
            print(f"Processing {len(seeds)} keywords automatically...")
            for kw in seeds:
                print(f"\n=== Processing keyword: {kw} ===")
                try:
                    process_recoll_fast(kw, max_results=recoll_limit, preview_chars=preview_chars)
                except Exception as e:
                    print(f"Error processing '{kw}': {e}")
                    traceback.print_exc()
        except ImportError:
            print("recoll_fast module not found. Please run the setup script.")
        return

    if server_mode:
        import uvicorn
        from server import app as server_app
        print(f"Starting OpenAI-compatible server on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
        uvicorn.run(server_app, host=config.SERVER_HOST, port=config.SERVER_PORT)
        return

    if build_recoll_index:
        if not input_path:
            print("Please provide --input <path> for building Recoll index.")
            return
        try:
            from core.file_utils import get_file_hash
            from extractors.registry import extract_text_from_file
            import recoll
            files = scan_files(input_path)
            print(f"Indexing {len(files)} files with Recoll...")
            db = recoll.connect(writable=True)
            for i, f in enumerate(files):
                print(f"  Indexing {i+1}/{len(files)}: {f.name}")
                try:
                    result = extract_text_from_file(f)
                    text = result.get("text", "")
                    if not text:
                        print("    No text extracted, skipping.")
                        continue
                    doc = recoll.Doc()
                    doc.url = f.as_uri()
                    doc.title = f.stem
                    doc.mimetype = result.get("format", "text/plain")
                    doc.text = text
                    file_hash = get_file_hash(str(f))
                    udi = f"thebrain:{file_hash}"
                    if db.needUpdate(udi, file_hash):
                        db.addOrUpdate(udi, doc)
                    else:
                        print("    Up to date, skipping.")
                except Exception as e:
                    print(f"    Recoll indexing error: {e}")
            db.close()
            print("Recoll index build complete.")
        except ImportError as e:
            print(f"Recoll not available: {e}")
        return
    if audit_mode:
        audit_all()
        return

    if chat_mode or deep_research:
        # Chat mode (optionally with deep research)
        from chat import analyze_query, retrieve_from_graph, fallback_to_chunks, build_context, generate_answer
        from chat.conversation import add_message, get_conversation_context
        from deep_research.coordinator import DeepResearchCoordinator

        print("Chat mode. Type 'exit' to quit. Add --deep-research to enable autonomous research.")
        session_id = f"cli_{int(time.time())}"
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
                    facts = retrieve_from_graph(analysis, max_depth=2)
                    if len(facts) < config.CHAT_MIN_FACTS_BEFORE_FALLBACK:
                        chunks = fallback_to_chunks(query)
                    else:
                        chunks = []
                    context = build_context(facts, chunks=chunks, conversation_history=conversation_history)
                    if logic_context:
                        context = logic_context + "\n\n" + context
                    if memory_text:
                        context = memory_text + "\n\n" + context
                    answer = generate_answer(query, context, conversation_history=conversation_history)

                add_message(session_id, "assistant", answer)
                print(f"Assistant:\n{answer}\n---")
            except KeyboardInterrupt:
                print("\nExiting chat.")
                break
        return

    if logic_mode and input_path:
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
        _recoll_interactive = "--recoll-interactive" in sys.argv
        if "--recoll-max-rounds" in sys.argv:
            _idx = sys.argv.index("--recoll-max-rounds") + 1
            if _idx < len(sys.argv):
                _recoll_max_rounds = int(sys.argv[_idx])
        run_recoll_guided_learning(process_file, tracker, max_rounds=_recoll_max_rounds, interactive=_recoll_interactive)
        return
    if guided:
        if not input_path:
            print("Please provide --input <path> for guided learning.")
            return
        input_path = Path(input_path)
        files = scan_files(input_path)
        # Progress bar support
        if config.USE_PROGRESS_BARS and config.TQDM_AVAILABLE:
            try:
                from tqdm import tqdm
                files_iter = tqdm(files, desc="Processing files", unit="file")
            except ImportError:
                files_iter = files
        else:
            files_iter = files
        total = len(files)
        tracker = ProgressTracker()
        tracker.total_files = total
        tracker.processed_count = 0
        try:
            if config.PARALLEL_PROCESSING_ENABLED:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import threading
                lock = threading.Lock()

                def process_with_lock(f):
                    with lock:
                        result = process_file(f, tracker, logic_context="")
                    return result

                with ThreadPoolExecutor(max_workers=config.PARALLEL_WORKERS) as executor:
                    futures = [executor.submit(process_with_lock, f) for f in files]
                    for future in as_completed(futures):
                        tracker.processed_count += 1
                        if future.result():
                            print("    (Success)")
            else:
                for f in files_iter:
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
                    success = process_file(f, tracker, logic_context=logic_context)
                    tracker.processed_count += 1
                    if success:
                        print("    (Success)")
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
