"""Trivium classifier proof (phase 86): stage routing + DB lookup + block."""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def test_trivium():
    from core.trivium import classify_stage, lookup_stage, trivium_context_block
    assert classify_stage("What is mitosis? Define the term.")[0] == "grammar"
    assert classify_stage("Why does mitosis happen? Compare mitosis and meiosis, what evidence?")[0] == "logic"
    assert classify_stage("Write a persuasive essay arguing for climate action")[0] == "rhetoric"
    s, c, hits = classify_stage("!!!")
    assert s == "grammar" and c == 0.0 and hits == []
    from core import db
    conn = db.db_connect("trivium")
    try:
        card = lookup_stage("logic", conn=conn)
        assert card and "syllogism" in card["description"].lower(), card
        assert lookup_stage("bogus", conn=conn) is None
        block = trivium_context_block("Why did Rome fall? What were the causes?", conn=conn)
        assert block.startswith("Trivium: processing at LOGIC"), block
        assert len(block) <= 600
        # extension content scale (phase 86 agents research)
        n_fal = conn.execute("SELECT COUNT(*) FROM entities WHERE entity_type='fallacy'").fetchone()[0]
        assert n_fal >= 50, n_fal
        n_soc = conn.execute("SELECT COUNT(*) FROM entities WHERE entity_type='socratic_stem'").fetchone()[0]
        assert n_soc >= 24, n_soc
        n_dev = conn.execute("SELECT COUNT(*) FROM entities WHERE entity_type='rhet_device'").fetchone()[0]
        assert n_dev >= 25, n_dev
        mp = conn.execute("SELECT description FROM entities WHERE canonical_id='trivium:form:modus-ponens'").fetchone()
        assert mp and "therefore" in mp[0].lower(), mp
        stem = conn.execute("SELECT description FROM entities WHERE type_family='assumption' AND entity_type='socratic_stem' LIMIT 1").fetchone()
        assert stem and "assum" in stem[0].lower(), stem
        gate = conn.execute("SELECT description FROM entities WHERE canonical_id='trivium:gate:evidence-grounded'").fetchone()
        assert gate and "citation" in gate[0].lower(), gate
    finally:
        conn.close()
    print("test_trivium: OK (routing, lookup, block)")


if __name__ == "__main__":
    test_trivium()
