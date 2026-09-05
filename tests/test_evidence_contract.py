"""Evidence-answer contract: builder shape + fixture validity."""
import json
from pathlib import Path
from retrieval.evidence_contract import build_contract, validate_contract

FIX = Path(__file__).parent.parent / "fixtures" / "evidence_answer"


def test_fixtures_valid():
    for name in ("technical-document.json", "vector-candidate.json"):
        doc = json.loads((FIX / name).read_text(encoding="utf-8"))
        assert validate_contract(doc) == [], name


def test_invalid_rejected():
    bad = {"contract": "evidence-answer/v1", "query": "q", "anchors": [],
           "claims": [{"claim_id": "c1", "statement": "s", "truth_class": "BOGUS",
                       "confidence": 0.5, "anchor_ids": [], "evidence": []}]}
    assert validate_contract(bad) != []


def test_builder_shape():
    dps = [{"type": "fact", "text": "Zealandia spans 4.9 Mkm2.", "confidence": 0.9,
            "doc_name": "z.pdf", "source_span": "4.9 Mkm2", "verification_status": "verified",
            "verification_layers": [{"verified": True}]}]
    doc = build_contract("How big is Zealandia?",
                         analysis={"keywords": ["Zealandia"], "entities": []},
                         datapoints=dps, chunks=[(3, 0.8)])
    assert validate_contract(doc) == []
    assert doc["claims"][0]["truth_class"] == "OBSERVED"
    assert doc["anchors"] and doc["vector_candidates"] == [{"chunk_id": 3, "similarity": 0.8}]
