"""Detector proof (phase 87): DB-driven patterns fire, clean text stays silent."""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def test_detect():
    from core import trivium_detect as D
    from core import db
    conn = db.db_connect("trivium")
    try:
        pats = D.load_patterns(conn)
        assert len(pats) >= 25, len(pats)
        # tu quoque fires with the right target
        hits = D.detect("You litter too, so you can not criticize my littering.", conn=conn)
        assert any(h[0] == "trivium:fal:tu-quoque" for h in hits), hits
        # bandwagon + authority fire
        hits2 = D.detect("Everyone agrees. Studies show it works.", conn=conn)
        ids2 = {h[0] for h in hits2}
        assert "trivium:fallacy:bandwagon" in ids2, hits2
        assert "trivium:fallacy:appeal-to-authority" in ids2, hits2
        # chiasmus structural fires
        hits3 = D.detect("Eat to live, not live to eat.", conn=conn)
        assert any(h[0] == "trivium:dev:chiasmus" for h in hits3), hits3
        # clean technical text stays silent
        hits4 = D.detect("The mixture was heated to 80 degrees for two hours.", conn=conn)
        assert hits4 == [], hits4
        # describe resolves a finding to its record
        card = D.describe(conn, "trivium:fal:tu-quoque")
        assert card and "accusing" in card["description"].lower(), card
        assert D.detect("", conn=conn) == []
    finally:
        conn.close()
    print("test_detect: OK (patterns fire, silence holds, records resolve)")


if __name__ == "__main__":
    test_detect()
