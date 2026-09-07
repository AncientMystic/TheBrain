"""
Nightly gazetteer export: mapping DB -> gazetteers/*.txt sidecars.

Writes mapping_*.txt files (never overwrites hand-curated base files):
mapping_cities.txt (geo entities), mapping_names.txt (person display
names + aliases), mapping_publishers.txt, mapping_works.txt. The
annotator picks these up automatically once wired (P4); until then they
are auditable artifacts proving population flows back to extraction.
Idempotent: full rewrite from DB each run.
"""
from pathlib import Path

EXPORTS = {
    "mapping_cities.txt": ("SELECT display_name FROM entities WHERE type_family='geo' AND entity_type IN ('city','town')", 0),
    "mapping_names.txt": ("SELECT display_name FROM entities WHERE type_family='person'", 0),
    "mapping_publishers.txt": ("SELECT display_name FROM entities WHERE entity_type='publisher'", 0),
    "mapping_works.txt": ("SELECT display_name FROM entities WHERE type_family='work' AND entity_type IN ('book','series')", 0),
    "mapping_aliases.txt": ("SELECT DISTINCT alias_norm FROM aliases", 0),
}


def export_gazetteers(conn, gaz_dir=None, min_len=2):
    """Write sidecar files from mapping DB. Returns {filename: count}."""
    import config as _cfg
    out_dir = Path(gaz_dir) if gaz_dir else Path(_cfg.GAZETTEERS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for fname, (sql, _) in EXPORTS.items():
        try:
            seen, terms = set(), []
            for (name,) in conn.execute(sql):
                t = str(name or "").strip()
                if len(t) < min_len or t.lower() in seen:
                    continue
                seen.add(t.lower())
                terms.append(t)
            terms.sort(key=str.lower)
            (out_dir / fname).write_text("\n".join(terms) + "\n", encoding="utf-8")
            counts[fname] = len(terms)
        except Exception as e:
            counts[fname] = f"ERR:{type(e).__name__}"
    try:
        conn.commit()
    except Exception:
        pass
    return counts


if __name__ == "__main__":
    from scripts.init_mapping_db import init_mapping_db
    _conn = init_mapping_db()
    _c = export_gazetteers(_conn)
    print(f"exported: {_c}")
    _conn.close()
