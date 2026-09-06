"""Aggregate braid: exact dedup (never merges distinct shapes) + loud outages."""
import itertools
from reasoning.beam import (
    canonical_claim_key, canonical_path_id, dedupe_beams, dedupe_claims,
    beam_search,
)


def _claim(s, p="related_to", o="Object", conf=0.8):
    return {"subject": s, "predicate": p, "object": o, "final_confidence": conf,
            "text": f"{s} {p} {o}"}


def test_key_normalizes_case_and_space():
    assert canonical_claim_key(_claim("  Alice ", "IS_A", "PERSON")) == ("alice", "is_a", "person")


def test_path_id_permutation_invariant():
    a = [_claim("Alice"), _claim("Bob"), _claim("Carol")]
    for perm in itertools.permutations(a):
        assert canonical_path_id(list(perm)) == canonical_path_id(a)


def test_path_id_distinguishes_shapes():
    assert canonical_path_id([_claim("Alice")]) != canonical_path_id([_claim("Bob")])


def test_dedupe_beams_keeps_best_stable():
    low = [_claim("Alice", conf=0.5), _claim("Bob", conf=0.5)]
    high = [_claim("Bob", conf=0.9), _claim("Alice", conf=0.9)]
    other = [_claim("Carol", conf=0.7)]
    out = dedupe_beams([low, other, high])
    assert len(out) == 2
    assert out[0] == high  # best score first, survivor of the isomorphic pair


def test_dedupe_claims_first_wins_and_empties_kept():
    dupes = [_claim("Alice", conf=0.5), _claim("alice", conf=0.9), {"text": "freeform"}]
    out = dedupe_claims(dupes)
    assert len(out) == 2
    assert out[0]["final_confidence"] == 0.5  # first occurrence kept, not rescored
    assert out[1] == {"text": "freeform"}


def test_diagnostics_default_off_compatible():
    # No diagnostics arg at all must behave exactly as before (empty in, empty out).
    best, all_claims = beam_search([], kg=object())
    assert best == [] and all_claims == []


def test_fallbacks_recorded_loudly():
    # Forced candidate-empty fallback: same inputs, diagnostics count the outage.
    from unittest.mock import patch
    import reasoning.beam as _beam
    sq = [{"question": "What is X?"}]
    fake_facts = [{"canonical_value": "X", "fact_type": "t", "fact_text": "X is Y.",
                   "source_span": "X is Y", "confidence": 0.9}]
    diag = {}
    with patch.object(_beam, "analyze_query", return_value={}), \
         patch.object(_beam, "retrieve_from_graph", return_value=fake_facts), \
         patch.object(_beam, "fallback_to_chunks", return_value=[]), \
         patch.object(_beam, "generate_candidate_claims", return_value=[]), \
         patch.object(_beam, "verify_claim_adaptive", side_effect=lambda c, **k: []):
        import config
        old = getattr(config, "ADAPTIVE_VERIFICATION", True)
        config.ADAPTIVE_VERIFICATION = True
        try:
            best, all_claims = _beam.beam_search(sq, kg=object(), beam_size=2, diagnostics=diag)
        finally:
            config.ADAPTIVE_VERIFICATION = old
    assert diag.get("candidate_empty_fallbacks", 0) >= 1
    assert best and all_claims  # fallback produced content, loudly recorded


def test_no_sub_questions_recorded():
    from unittest.mock import patch
    import reasoning.beam as _beam
    diag = {}
    with patch.object(_beam, "decompose_query", return_value=[]):
        best, all_claims = _beam.reason_with_verification("x", kg=object(), diagnostics=diag)
    assert best == [] and all_claims == []
    assert diag.get("no_sub_questions") == 1
