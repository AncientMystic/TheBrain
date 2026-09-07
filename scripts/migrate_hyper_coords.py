"""
Hyper-coordinate columns migration (phase 1 of hyper-placement geometry).

Adds lat/lon/depth/shard/octonion/E8 columns to mapping entities plus the
shards table. Idempotent (checks pragma before each ALTER); additive-only,
never rewrites existing data. Run: scripts/migrate_hyper_coords.py [db-path]
(defaults to live mapping.db).
"""
import sqlite3
import sys

COLUMNS = [
    ("lat_str", "TEXT"),
    ("lon_str", "TEXT"),
    ("lat_r", "REAL"),
    ("lon_r", "REAL"),
    ("depth", "REAL"),
    ("shard_key", "TEXT"),
    ("oct8", "BLOB"),
    ("e8_key", "TEXT"),
]

SHARDS_SQL = """
CREATE TABLE IF NOT EXISTS shards(
    shard_key TEXT PRIMARY KEY,
    centroid BLOB,
    u1 BLOB, u2 BLOB,
    k REAL, k2 REAL, dmax REAL, radius REAL,
    home_e8 TEXT,
    affinity_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_ent_shard_ll ON entities(shard_key, lat_r, lon_r);
CREATE INDEX IF NOT EXISTS idx_ent_e8 ON entities(e8_key);
"""

def migrate(conn):
    """Apply migration. Returns list of added columns (for logging/tests)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(entities)")}
    added = []
    for name, typ in COLUMNS:
        if name not in cols and name != "shard_key":
            conn.execute(f"ALTER TABLE entities ADD COLUMN {name} {typ}")
            added.append(name)
    # shard_key may predate (added in an earlier phase); ensure it too.
    if "shard_key" not in cols:
        conn.execute("ALTER TABLE entities ADD COLUMN shard_key TEXT")
        added.append("shard_key")
    conn.executescript(SHARDS_SQL)
    try:
        scols = {r[1] for r in conn.execute("PRAGMA table_info(shards)")}
        if "k2" not in scols:
            conn.execute("ALTER TABLE shards ADD COLUMN k2 REAL")
            added.append("shards.k2")
    except Exception:
        pass
    conn.commit()
    return added


def verify(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(entities)")}
    need = {n for n, _ in COLUMNS} | {"shard_key"}
    assert need <= cols, f"missing: {need - cols}"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "shards" in tables
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _conn = sqlite3.connect(sys.argv[1])
    else:
        sys.path.insert(0, "A:/scripts/TheBrain")
        from core import db as _db
        _conn = _db.db_connect("mapping")
    _added = migrate(_conn)
    verify(_conn)
    print(f"migration ok; added: {_added or 'nothing (already current)'}")
    _conn.close()
