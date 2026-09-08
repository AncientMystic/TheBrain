"""Docs ETL (phase 89): trivium programming rows -> language sphere tables.

Routes by TAG family from populate_trivium4.ROWS: GRAMMAR_<LANG> to that
language sphere (stage grammar), UNIV_* to lang_universal, LOGIC/RHET/STD/
SOCRATIC to lang_practices (stage logic/rhetoric/standard/review).
Idempotent (topic PK REPLACE). Run: etl_programming.py.
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")
sys.path.insert(0, "A:/scripts/TheBrain/scripts")

LANG_OF = {"GRAMMAR_PYTHON": ("lang_python", "grammar"),
           "GRAMMAR_JS": ("lang_javascript", "grammar"),
           "GRAMMAR_SQL": ("lang_sql", "grammar"),
           "GRAMMAR_SYSTEMS": ("lang_systems", "grammar"),
           "GRAMMAR_WEB": ("lang_web", "grammar")}
AREA_OF = {"UNIV": ("lang_universal", None), "LOGIC": ("lang_practices", None),
           "RHET": ("lang_practices", None), "STD": ("lang_practices", None),
           "SOCRATIC": ("lang_practices", "review")}
STAGE_OF = {"GRAMMAR": "grammar", "UNIV": "grammar", "LOGIC": "logic",
            "RHET": "rhetoric", "STD": "standard", "SOCRATIC": "review"}


def run():
    import time
    from core import db as _db
    from populate_trivium4 import ROWS
    tc = _db.db_connect("trivium")
    pc = _db.db_connect("programming")
    n = 0
    for tag, dn, desc, als in ROWS:
        fam = tag.split("_", 1)[0]
        if tag in LANG_OF:
            table, stage = LANG_OF[tag]
        elif fam in AREA_OF:
            table, _ = AREA_OF[fam]
            stage = STAGE_OF.get(fam, "grammar")
        else:
            continue
        row = tc.execute("SELECT canonical_id FROM entities WHERE display_name=?",
                         (dn,)).fetchone()
        ref = row[0] if row else ""
        pc.execute(f"INSERT OR REPLACE INTO {table}"
                   "(topic, stage, rule, example, trivium_ref, updated_at)"
                   " VALUES (?,?,?,?,?,?)",
                   (dn, stage, desc, "", ref, int(time.time())))
        n += 1
    # fallacy-link notes -> lang_practices as review aids
    for row in tc.execute("SELECT display_name, description, canonical_id FROM entities"
                          " WHERE entity_type='fallacy_link'"):
        pc.execute("INSERT OR REPLACE INTO lang_practices"
                   "(topic, stage, rule, example, trivium_ref, updated_at)"
                   " VALUES (?,?,?,?,?,?)",
                   (row[0], "review", row[1], "", row[2], int(time.time())))
        n += 1
    pc.commit()
    counts = {}
    for t in ("lang_python", "lang_javascript", "lang_sql", "lang_systems",
              "lang_web", "lang_universal", "lang_practices"):
        counts[t] = pc.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    tc.close()
    pc.close()
    return n, counts


if __name__ == "__main__":
    _n, _c = run()
    print(f"etl_programming: {_n} topics; { _c}")
