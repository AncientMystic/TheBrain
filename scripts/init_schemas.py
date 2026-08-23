import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import db
from scripts.init_recoll_log_db import init_recoll_log_db


def init_index_db():
    conn = db.db_connect("index")
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS documents (
        file_hash TEXT PRIMARY KEY, file_path TEXT NOT NULL, filename TEXT NOT NULL,
        file_format TEXT NOT NULL, title TEXT, author TEXT, year TEXT,
        page_count INTEGER, text_length INTEGER, ocr_used INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS document_chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL,
        chunk_index INTEGER NOT NULL, chunk_text TEXT NOT NULL,
        start_offset INTEGER, end_offset INTEGER,
        FOREIGN KEY(doc_hash) REFERENCES documents(file_hash))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS processing_progress (
        file_hash TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'processed',
        stage TEXT, updated_at TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS llm_extraction_cache (
        chunk_hash TEXT, category TEXT, model TEXT, max_tokens INTEGER,
        result_json TEXT, prompt_hash TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (chunk_hash, category, model, max_tokens, prompt_hash))""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_format ON documents(file_format)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_hash ON document_chunks(doc_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_index ON document_chunks(chunk_index)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_chunk_hash ON llm_extraction_cache(chunk_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_category ON llm_extraction_cache(category)")
    conn.commit(); conn.close()
    print("[init] index.db ready")


def init_summaries_db():
    conn = db.db_connect("summaries"); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS doc_summaries (
        doc_hash TEXT PRIMARY KEY, doc_name TEXT NOT NULL, summary TEXT,
        key_points_json TEXT, summary_embedding BLOB, verification_status TEXT DEFAULT 'unverified')""")
    cur.execute("""CREATE TABLE IF NOT EXISTS summary_versions (
        version_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL,
        summary TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_summaries_doc_hash ON doc_summaries(doc_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_summaries_doc_name ON doc_summaries(doc_name)")
    conn.commit(); conn.close()
    print("[init] summaries.db ready")


def init_key_facts_db():
    conn = db.db_connect("key_facts"); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS key_facts (
        fact_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, doc_name TEXT NOT NULL,
        fact_type TEXT NOT NULL, fact_text TEXT NOT NULL, canonical_value TEXT,
        source_span TEXT, confidence REAL DEFAULT 0.0, verified INTEGER DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS entities (
        entity_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, entity_type TEXT NOT NULL,
        entity_name TEXT NOT NULL, normalized_name TEXT, source_span TEXT, confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS people (
        person_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, person_name TEXT NOT NULL,
        normalized_name TEXT, role TEXT, source_span TEXT, confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS locations (
        location_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, location_name TEXT NOT NULL,
        normalized_place TEXT, country TEXT, admin1 TEXT, location_type TEXT, source_span TEXT, confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS dates (
        date_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, date_text TEXT NOT NULL,
        normalized_date TEXT, date_type TEXT, source_span TEXT, confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, event_name TEXT NOT NULL,
        normalized_name TEXT, event_date TEXT, event_type TEXT, description TEXT, significance TEXT,
        source_span TEXT, confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS discoveries (
        discovery_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, discovery_name TEXT NOT NULL,
        normalized_name TEXT, description TEXT, date TEXT, significance TEXT, source_span TEXT, confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS gems (
        gem_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, gem_text TEXT NOT NULL,
        category TEXT, importance REAL DEFAULT 0.0, source_span TEXT, confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS fact_sources (
        source_id INTEGER PRIMARY KEY AUTOINCREMENT, fact_id INTEGER NOT NULL, doc_hash TEXT NOT NULL,
        chunk_id INTEGER, evidence_span TEXT, exact_quote TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS entity_fact_index (
        fact_id INTEGER NOT NULL,
        entity_name TEXT,
        normalized_name TEXT,
        PRIMARY KEY (fact_id, entity_name)
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_fact_index_norm ON entity_fact_index(normalized_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_entity_fact_index_fact ON entity_fact_index(fact_id)")
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS key_facts_fts USING fts5(fact_text, canonical_value, source_span)")
    # Triggers to keep FTS in sync
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS key_facts_ai AFTER INSERT ON key_facts BEGIN
            INSERT INTO key_facts_fts(rowid, fact_text, canonical_value, source_span)
            VALUES (new.fact_id, new.fact_text, new.canonical_value, new.source_span);
        END;
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS key_facts_ad AFTER DELETE ON key_facts BEGIN
            DELETE FROM key_facts_fts WHERE rowid = old.fact_id;
        END;
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS key_facts_au AFTER UPDATE ON key_facts BEGIN
            DELETE FROM key_facts_fts WHERE rowid = old.fact_id;
            INSERT INTO key_facts_fts(rowid, fact_text, canonical_value, source_span)
            VALUES (new.fact_id, new.fact_text, new.canonical_value, new.source_span);
        END;
    """)

    for table, col in [("key_facts","doc_hash"),("key_facts","fact_type"),("entities","doc_hash"),("entities","entity_type"),
                       ("people","doc_hash"),("people","normalized_name"),("locations","doc_hash"),("locations","normalized_place"),
                       ("dates","doc_hash"),("dates","normalized_date"),("events","doc_hash"),("events","normalized_name"),
                       ("discoveries","doc_hash"),("discoveries","normalized_name"),("gems","doc_hash"),("gems","category"),
                       ("fact_sources","doc_hash"),("fact_sources","fact_id")]:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col})")
    conn.commit(); conn.close()
    print("[init] key_facts.db ready")


def init_embeddings_db():
    conn = db.db_connect("embeddings"); cur = conn.cursor()
    # Migration: drop old embedding_cache if it has single-column PK
    cur.execute("PRAGMA table_info(embedding_cache)")
    cols = [row[1] for row in cur.fetchall()]
    if cols and "model" not in cols:
        cur.execute("DROP TABLE embedding_cache")
    cur.execute("""CREATE TABLE IF NOT EXISTS document_embeddings (
        doc_hash TEXT PRIMARY KEY, embedding BLOB NOT NULL, model TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS chunk_embeddings (
        chunk_id INTEGER PRIMARY KEY, doc_hash TEXT NOT NULL, chunk_text TEXT NOT NULL,
        embedding BLOB NOT NULL, model TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS embedding_cache (
        text TEXT NOT NULL, embedding BLOB NOT NULL, model TEXT NOT NULL,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (text, model))""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc_hash ON document_embeddings(doc_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_doc_hash ON chunk_embeddings(doc_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_chunk_id ON chunk_embeddings(chunk_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(model)")
    conn.commit(); conn.close()
    print("[init] embeddings.db ready")


def init_hypergraph_db():
    conn = db.db_connect("hypergraph"); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS nodes (
        node_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, node_type TEXT NOT NULL,
        node_text TEXT NOT NULL, normalized_name TEXT, attributes_json TEXT, source_span TEXT, confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, source_node_id INTEGER NOT NULL,
        target_node_id INTEGER NOT NULL, relation_type TEXT NOT NULL, weight REAL DEFAULT 1.0,
        evidence_span TEXT, confidence REAL DEFAULT 0.0,
        FOREIGN KEY(source_node_id) REFERENCES nodes(node_id),
        FOREIGN KEY(target_node_id) REFERENCES nodes(node_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS doc_entity_nodes (
        doc_hash TEXT NOT NULL, entity_type TEXT NOT NULL, entity_name TEXT NOT NULL, node_id INTEGER,
        PRIMARY KEY(doc_hash, entity_type, entity_name),
        FOREIGN KEY(node_id) REFERENCES nodes(node_id))""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_doc_hash ON nodes(doc_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_normalized_name ON nodes(normalized_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_doc_hash ON edges(doc_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id)")
    conn.commit(); conn.close()
    print("[init] hypergraph.db ready")


def init_external_graph_db():
    conn = db.db_connect("external_graph"); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS global_nodes (
        global_node_id INTEGER PRIMARY KEY AUTOINCREMENT, canonical_name TEXT NOT NULL, node_type TEXT NOT NULL,
        aliases_json TEXT, attributes_json TEXT, embedding BLOB, cluster_id INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS global_edges (
        edge_id INTEGER PRIMARY KEY AUTOINCREMENT, source_node_id INTEGER NOT NULL, target_node_id INTEGER NOT NULL,
        relation_type TEXT NOT NULL, weight REAL DEFAULT 1.0, doc_hash TEXT, source_span TEXT,
        confidence REAL DEFAULT 0.0, occurrence_count INTEGER DEFAULT 1,
        FOREIGN KEY(source_node_id) REFERENCES global_nodes(global_node_id),
        FOREIGN KEY(target_node_id) REFERENCES global_nodes(global_node_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS topic_nodes (
        topic_id INTEGER PRIMARY KEY AUTOINCREMENT, topic_name TEXT NOT NULL UNIQUE, category TEXT, keywords_json TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS keyword_topic_edges (
        keyword TEXT NOT NULL, topic TEXT NOT NULL, weight REAL DEFAULT 1.0, PRIMARY KEY (keyword, topic))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS keyword_cooccurrence (
        kw_a TEXT NOT NULL, kw_b TEXT NOT NULL, weight REAL DEFAULT 1.0, doc_hashes TEXT, PRIMARY KEY (kw_a, kw_b))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS cross_doc_links (
        link_id INTEGER PRIMARY KEY AUTOINCREMENT, doc_hash TEXT NOT NULL, global_node_id INTEGER NOT NULL,
        weight REAL DEFAULT 1.0, relation_type TEXT,
        FOREIGN KEY(global_node_id) REFERENCES global_nodes(global_node_id))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS topic_hierarchy (
        parent TEXT NOT NULL, child TEXT NOT NULL, weight REAL DEFAULT 1.0, PRIMARY KEY (parent, child))""")
    cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS global_nodes_fts USING fts5(canonical_name, node_type)")
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS global_nodes_fts_ai AFTER INSERT ON global_nodes BEGIN
            INSERT INTO global_nodes_fts(rowid, canonical_name, node_type)
            VALUES (new.global_node_id, new.canonical_name, new.node_type);
        END;
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS global_nodes_fts_ad AFTER DELETE ON global_nodes BEGIN
            DELETE FROM global_nodes_fts WHERE rowid = old.global_node_id;
        END;
    """)
    cur.execute("""
        CREATE TRIGGER IF NOT EXISTS global_nodes_fts_au AFTER UPDATE ON global_nodes BEGIN
            DELETE FROM global_nodes_fts WHERE rowid = old.global_node_id;
            INSERT INTO global_nodes_fts(rowid, canonical_name, node_type)
            VALUES (new.global_node_id, new.canonical_name, new.node_type);
        END;
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_global_nodes_name ON global_nodes(canonical_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_global_nodes_type ON global_nodes(node_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_global_edges_source ON global_edges(source_node_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_global_edges_target ON global_edges(target_node_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_keyword_topic_keyword ON keyword_topic_edges(keyword)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_keyword_topic_topic ON keyword_topic_edges(topic)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_keyword_cooccur_a ON keyword_cooccurrence(kw_a)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_keyword_cooccur_b ON keyword_cooccurrence(kw_b)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cross_doc_doc_hash ON cross_doc_links(doc_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cross_doc_node ON cross_doc_links(global_node_id)")
    conn.commit(); conn.close()
    print("[init] external-graph.db ready")


def init_ocr_db():
    conn = db.db_connect("ocr"); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS ocr_cache (
        file_hash TEXT, pages INTEGER, dpi INTEGER, ocr_text TEXT, PRIMARY KEY (file_hash, pages, dpi))""")
    conn.commit(); conn.close()
    print("[init] ocr_cache.db ready")


def init_memories_db():
    conn = db.db_connect("memories"); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS memory_entries (
        memory_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        memory_type TEXT, content TEXT NOT NULL, importance REAL DEFAULT 0.5, embedding BLOB, verified INTEGER DEFAULT 1)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS memory_keywords (
        memory_id INTEGER REFERENCES memory_entries(memory_id), keyword TEXT, weight REAL DEFAULT 1.0,
        PRIMARY KEY (memory_id, keyword))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS memory_sessions (
        session_id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP, last_active TEXT)""")
    conn.commit(); conn.close()
    print("[init] memories.db ready")


def init_logic_db():
    conn = db.db_connect("logic"); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS logic_modules (
        logic_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, category TEXT, description TEXT,
        summary TEXT, keywords TEXT, content TEXT, embedding BLOB, verified INTEGER DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS logic_examples (
        example_id INTEGER PRIMARY KEY AUTOINCREMENT, logic_id INTEGER REFERENCES logic_modules(logic_id),
        input_text TEXT, output_text TEXT, source_span TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS logic_keywords (
        logic_id INTEGER REFERENCES logic_modules(logic_id), keyword TEXT, weight REAL DEFAULT 1.0,
        PRIMARY KEY (logic_id, keyword))""")
    cur.execute("""CREATE TABLE IF NOT EXISTS logic_tags (
        logic_id INTEGER REFERENCES logic_modules(logic_id), tag TEXT)""")
    conn.commit(); conn.close()
    print("[init] logic.db ready")


def init_reasoning_db():
    conn = db.db_connect("reasoning"); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS research_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        research_id TEXT NOT NULL,
        node_type TEXT NOT NULL,
        name TEXT NOT NULL,
        content TEXT,
        confidence REAL DEFAULT 0.0,
        metadata_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS research_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        research_id TEXT NOT NULL,
        source_node_id INTEGER REFERENCES research_nodes(id),
        target_node_id INTEGER REFERENCES research_nodes(id),
        relation_type TEXT,
        weight REAL DEFAULT 1.0,
        evidence TEXT,
        confidence REAL DEFAULT 0.0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS reasoning_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, query_id TEXT, step_number INTEGER, node_type TEXT,
        content TEXT, formal_representation TEXT, confidence REAL, status TEXT DEFAULT 'pending',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS reasoning_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT, source_node_id INTEGER REFERENCES reasoning_nodes(id),
        target_node_id INTEGER REFERENCES reasoning_nodes(id), relation_type TEXT, verified INTEGER DEFAULT 0)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS grounding_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT, reasoning_node_id INTEGER REFERENCES reasoning_nodes(id),
        grounding_type TEXT, kg_triple_id INTEGER, text_span_id INTEGER, prior_step_id INTEGER, confidence REAL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS kg_triples (
        id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, predicate TEXT, object TEXT,
        source_document_id INTEGER, confidence REAL, verification_status TEXT DEFAULT 'unverified')""")
    cur.execute("""CREATE TABLE IF NOT EXISTS reasoning_paths (
        id INTEGER PRIMARY KEY AUTOINCREMENT, query_id TEXT, path TEXT, final_answer TEXT,
        confidence REAL, verification_summary TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS implied_triples (
        subject TEXT, predicate TEXT, object TEXT,
        PRIMARY KEY (subject, predicate, object))""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_implied_subject_pred ON implied_triples(subject, predicate)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_implied_object ON implied_triples(object)")
    cur.execute("""CREATE TABLE IF NOT EXISTS reasoning_dependencies (
        id INTEGER PRIMARY KEY AUTOINCREMENT, step_id INTEGER REFERENCES reasoning_nodes(id),
        depends_on_id INTEGER REFERENCES reasoning_nodes(id), dependency_type TEXT, verified BOOLEAN)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS verification_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, step_id INTEGER REFERENCES reasoning_nodes(id),
        layer TEXT, verified BOOLEAN, confidence REAL, details TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS contradiction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, triple_a_id INTEGER, triple_b_id INTEGER,
        status TEXT, resolved_by TEXT, resolved_at TEXT,
        details TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS agent_actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, agent_name TEXT, action TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
        details TEXT)""")
    conn.commit(); conn.close()
    print("[init] reasoning.db ready")


def init_verification_standards_db():
    conn = db.db_connect("verification_standards")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verified_standards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            standard_id TEXT UNIQUE,
            statement TEXT NOT NULL,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            negation INTEGER DEFAULT 0,
            temporal_bounds_json TEXT,
            truth_status TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_doc_hash TEXT,
            source_span TEXT,
            priority INTEGER DEFAULT 2,
            confidence REAL DEFAULT 0.0,
            source_hierarchy_level INTEGER,
            data_model_policy TEXT,
            psych_score_total INTEGER,
            psych_score_breakdown_json TEXT,
            enforcement_vector TEXT,
            intentionality_triad_json TEXT,
            lived_experience_cluster INTEGER DEFAULT 0,
            funding_gatekeeping_flags_json TEXT,
            socratic_assessment_json TEXT,
            supporting_evidence_json TEXT,
            provenance_json TEXT,
            verified_by TEXT,
            verified_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verified_standard_sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_doc_hash TEXT UNIQUE,
            file_path TEXT,
            title TEXT,
            source_hierarchy_level INTEGER,
            socratic_assessment_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS standard_comparisons (
            comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_type TEXT NOT NULL,
            fact_id INTEGER NOT NULL,
            standard_id INTEGER,
            relation TEXT,
            method TEXT,
            confidence REAL,
            rationale TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS verification_promotions (
            doc_hash TEXT PRIMARY KEY,
            promoted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT,
            standards_inserted INTEGER DEFAULT 0
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_standards_statement ON verified_standards(statement)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_standards_subject_predicate ON verified_standards(subject, predicate, object)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_standards_source_type ON verified_standards(source_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_standards_truth_status ON verified_standards(truth_status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_comparisons_fact ON standard_comparisons(fact_type, fact_id)")
    conn.commit()
    conn.close()
    print("[init] verification_standards.db ready")


def _alter_table_if_needed(conn, table, column, definition):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(%s)" % table)
    cols = [row[1] for row in cur.fetchall()]
    if column not in cols:
        cur.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, definition))
        conn.commit()

def apply_existing_table_migrations():
    """Add verification_status columns and temporal/negation columns."""
    # key_facts
    conn = db.db_connect("key_facts")
    _alter_table_if_needed(conn, "key_facts", "verification_status", "TEXT DEFAULT 'unverified'")
    _alter_table_if_needed(conn, "key_facts", "verified_by", "TEXT")
    _alter_table_if_needed(conn, "key_facts", "verified_at", "TEXT")
    _alter_table_if_needed(conn, "key_facts", "negation", "INTEGER DEFAULT 0")
    _alter_table_if_needed(conn, "key_facts", "valid_from", "TEXT")
    _alter_table_if_needed(conn, "key_facts", "valid_to", "TEXT")
    conn.close()

    # documents
    conn = db.db_connect("index")
    _alter_table_if_needed(conn, "documents", "is_verified_source", "INTEGER DEFAULT 0")
    conn.close()

    # global_edges
    conn = db.db_connect("external_graph")
    _alter_table_if_needed(conn, "global_edges", "verification_status", "TEXT DEFAULT 'unverified'")
    _alter_table_if_needed(conn, "global_edges", "verified_by", "TEXT")
    _alter_table_if_needed(conn, "global_edges", "verified_at", "TEXT")
    _alter_table_if_needed(conn, "global_edges", "negation", "INTEGER DEFAULT 0")
    _alter_table_if_needed(conn, "global_edges", "valid_from", "TEXT")
    _alter_table_if_needed(conn, "global_edges", "valid_to", "TEXT")
    conn.close()

    # kg_triples
    conn = db.db_connect("reasoning")
    _alter_table_if_needed(conn, "kg_triples", "verification_status", "TEXT DEFAULT 'unverified'")
    _alter_table_if_needed(conn, "kg_triples", "verified_by", "TEXT")
    _alter_table_if_needed(conn, "kg_triples", "verified_at", "TEXT")
    _alter_table_if_needed(conn, "kg_triples", "negation", "INTEGER DEFAULT 0")
    _alter_table_if_needed(conn, "kg_triples", "valid_from", "TEXT")
    _alter_table_if_needed(conn, "kg_triples", "valid_to", "TEXT")
    conn.close()
    print("[init] verification_status columns added to existing tables")


def init_all():
    print("Initializing SQLite databases...")
    init_index_db()
    init_summaries_db()
    init_key_facts_db()
    init_embeddings_db()
    init_hypergraph_db()
    init_external_graph_db()
    init_ocr_db()
    init_memories_db()
    init_logic_db()
    init_reasoning_db()
    init_recoll_log_db()
    init_verification_standards_db()
    apply_existing_table_migrations()
    print("All databases ready.")


if __name__ == "__main__":
    init_all()
