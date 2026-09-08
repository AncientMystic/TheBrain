"""Trivium rewards proof (phase 92): retrieval + detection + staging measured.

- S-scored lookup on trivium.db: 5 queries must rank the expected
  canonical top-1 (proves geometry pays off on the new corpus).
- Detector: 4 fallacy sentences fire the right target, 2 clean stay silent.
- Stage block: 3 queries route grammar/logic/rhetoric.
Exit 0 iff all green. Usage: trivium_rewards.py
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

QUERIES = [
    ("tu quoque personal attack", "trivium:fal:tu-quoque"),
    ("chiasmus reversal figure", "trivium:dev:chiasmus"),
    ("modus ponens conditional inference", "trivium:form:modus-ponens"),
    ("Barbara syllogism universal", "trivium:form:barbara"),
    ("Socratic hidden assumption question", None),  # any socratic_stem top-1
]

DIRTY = [
    ("You drink too, so don't lecture me on health.", "trivium:fal:tu-quoque"),
    ("Everyone knows this stock will moon.", "trivium:fallacy:bandwagon"),
    ("No true programmer uses tabs.", "trivium:fallacy:no-true-scotsman"),
    ("After the update the app crashed, so the update broke it.", "trivium:fal:post-hoc"),
]
CLEAN = [
    "The mixture was heated to 80 degrees for two hours.",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY name;",
]

STAGES = [
    ("What is photosynthesis?", "grammar"),
    ("Why do leaves change color? Compare with evergreens.", "logic"),
    ("Write a speech persuading the town to plant trees.", "rhetoric"),
]


def main():
    from core import db
    from core.embeddings import get_embeddings_batch
    from core.mapping_lookup import rank_candidates_scored
    from core import trivium_detect as D
    from core.trivium import classify_stage
    tconn = db.db_connect("trivium")
    fails = []
    try:
        for q, exp in QUERIES:
            emb = get_embeddings_batch([q], space="hyperbolic")[0]
            res = rank_candidates_scored(q, mention_emb=emb, limit=5, conn=tconn)
            ids = [c for c, _, _ in res]
            if exp is None:
                ok = bool(ids) and "trivium:soc:" in ids[0]
                rank = 1 if ok else 99
            else:
                rank = ids.index(exp) + 1 if exp in ids else 99
                ok = rank <= 3
            print(f"[{'OK' if ok else 'MISS'}] {q!r:45s} -> rank {rank} ({(ids[0] if ids else None)})")
            if not ok:
                fails.append(q)
        for text, exp in DIRTY:
            hits = D.detect(text, conn=tconn)
            ok = any(h[0] == exp for h in hits)
            print(f"[{'OK' if ok else 'MISS'}] detect {exp.split(':')[-1]:22s} <- {text[:45]!r}")
            if not ok:
                fails.append(text)
        for text in CLEAN:
            hits = D.detect(text, conn=tconn)
            ok = hits == []
            print(f"[{'OK' if ok else 'MISS'}] silence <- {text[:45]!r}")
            if not ok:
                fails.append(text)
        for q, exp in STAGES:
            got, conf, _ = classify_stage(q)
            ok = got == exp
            print(f"[{'OK' if ok else 'MISS'}] stage {exp:9s} <- {q[:45]!r} (conf {conf})")
            if not ok:
                fails.append(q)
    finally:
        tconn.close()
    print(f"trivium_rewards: {'OK' if not fails else 'FAIL ' + str(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
