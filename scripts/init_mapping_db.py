"""
Mapping-database schema + initializer (P2 foundation).

One logical DB: central router (mapping.db) + per-shard files
(mapping_{shard}.db) sharing this identical DDL. Shard routing key:
{family}:{region} (e.g. geo:eu, person:global, work:global).

Uses the existing pooled db_connect("mapping") for the live path; tests
call init_mapping_db(explicit_path) against temp files instead.
"""
import json
import sqlite3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    canonical_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    type_family TEXT NOT NULL,
    display_name TEXT NOT NULL,
    shard_key TEXT NOT NULL,
    region_code TEXT,
    description TEXT NOT NULL DEFAULT '',
    popularity REAL DEFAULT 0,
    external_ids TEXT DEFAULT '{}',
    centroid_version INT DEFAULT 1,
    updated_at INT DEFAULT 0,
    source TEXT DEFAULT '',
    emb BLOB
);
CREATE INDEX IF NOT EXISTS idx_ent_type_region ON entities(entity_type, region_code);
CREATE INDEX IF NOT EXISTS idx_ent_shard ON entities(shard_key);
CREATE INDEX IF NOT EXISTS idx_ent_pop ON entities(popularity DESC);

CREATE TABLE IF NOT EXISTS aliases (
    alias_norm TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    lang TEXT DEFAULT 'en',
    alias_type TEXT DEFAULT 'aka',
    PRIMARY KEY (alias_norm, canonical_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_alias_norm ON aliases(alias_norm);

CREATE VIRTUAL TABLE IF NOT EXISTS aliases_fts USING fts5(alias_norm, canonical_id UNINDEXED);

CREATE TABLE IF NOT EXISTS entity_relations (
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    rel TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, rel)
);
CREATE INDEX IF NOT EXISTS idx_rel_dst ON entity_relations(dst_id, rel);

CREATE TABLE IF NOT EXISTS shard_centroids (
    shard_key TEXT PRIMARY KEY,
    shard_file TEXT NOT NULL,
    centroid BLOB NOT NULL,
    count INT DEFAULT 0,
    updated_at INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ingest_provenance (
    canonical_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    fetched_at INT DEFAULT 0,
    PRIMARY KEY (canonical_id, source, source_id)
);

CREATE TABLE IF NOT EXISTS dirty_mentions (
    surface TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    proposed_canonical TEXT DEFAULT '',
    status TEXT DEFAULT 'pending',
    PRIMARY KEY (surface, doc_id)
);

CREATE TABLE IF NOT EXISTS terms (
    term TEXT PRIMARY KEY,
    kinds TEXT NOT NULL DEFAULT '[]',
    risk TEXT DEFAULT 'low',
    region_hint TEXT,
    requires_coevidence TEXT NOT NULL DEFAULT '[]',
    negative_coevidence TEXT NOT NULL DEFAULT '[]',
    confidence_penalty REAL DEFAULT 0,
    default_kind TEXT,
    action TEXT DEFAULT 'keep',
    notes TEXT DEFAULT ''
);
"""


def init_mapping_db(path=None):
    """Create (or verify) the mapping schema. Returns the sqlite3 connection.

    path: explicit file (tests); None = live mapping.db via pooled db_connect.
    """
    if path is None:
        from core import db as _db
        conn = _db.db_connect("mapping")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        return conn
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def import_disambiguation_notes(conn, notes_path=None):
    """Seed the terms table from gazetteers/disambiguation_notes.json.

    Idempotent (INSERT OR REPLACE). Returns number of rows written.
    """
    import config as _cfg
    from pathlib import Path as _P
    src = _P(notes_path) if notes_path else _P(_cfg.GAZETTEERS_DIR) / "disambiguation_notes.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    for n in data.get("notes", []):
        rows.append((
            str(n.get("term", "")).strip().lower(),
            json.dumps(n.get("kinds", [])),
            str(n.get("risk", "low")),
            n.get("region_hint"),
            json.dumps(n.get("requires_coevidence", [])),
            json.dumps(n.get("negative_coevidence", [])),
            float(n.get("confidence_penalty", 0) or 0),
            n.get("default_kind"),
            str(n.get("action", "keep")),
            str(n.get("notes", "")),
        ))
    rows = [r for r in rows if r[0]]
    conn.executemany(
        """INSERT OR REPLACE INTO terms
           (term, kinds, risk, region_hint, requires_coevidence,
            negative_coevidence, confidence_penalty, default_kind, action, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return len(rows)


if __name__ == "__main__":
    _c = init_mapping_db()
    _n = import_disambiguation_notes(_c)
    print(f"mapping.db ready; { _n} disambiguation terms imported.")
    _c.close()
