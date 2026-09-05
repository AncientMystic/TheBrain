"""Linguistic primitives: fallback-identity first, then shape guarantees."""
from extraction.nlp_primitives import split_sentences, pos_filtered_entities


def test_splitter_abbreviations_and_decimals():
    text = "Dr. Alice went to Paris. The value was 4.9 in Jan. 2020. She stayed."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0][0].startswith("Dr. Alice")
    assert "4.9" in sents[1][0]


def test_splitter_offsets_recoverable():
    text = "First sentence. Second one here."
    for s, start, end in split_sentences(text):
        assert text[start:end] == s


def test_splitter_empty_safe():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_pos_filter_fallback_identity():
    # Backend unavailable: input must pass through byte-identical.
    import extraction.nlp_primitives as _np
    old_nlp, old_tried = _np._SPACY_NLP, _np._SPACY_TRIED
    _np._SPACY_NLP, _np._SPACY_TRIED = None, True
    try:
        assert _np._spacy_doc("anything at all") is None
        ents = [{"type": "PERSON", "text": "Marie Curie", "confidence": 0.8},
                {"type": "ORG", "text": "Running Club", "confidence": 0.7}]
        assert pos_filtered_entities(ents, "Paris Running Club met") == ents
    finally:
        _np._SPACY_NLP, _np._SPACY_TRIED = old_nlp, old_tried


def test_pos_filter_real_backend():
    # en_core_web_sm installed: verb-headed ORG dropped, PROPN person kept.
    import extraction.nlp_primitives as _np
    assert _np._spacy_doc("Marie Curie discovered radium.") is not None
    ents = [{"type": "PERSON", "text": "Marie Curie", "confidence": 0.9},
            {"type": "ORG", "text": "Running Fast", "confidence": 0.7}]
    kept = pos_filtered_entities(ents, "Marie Curie discovered radium. Running Fast happened.")
    texts = [e["text"] for e in kept]
    assert "Marie Curie" in texts
    assert "Running Fast" not in texts


def test_event_window_sentence_edges():
    from extraction.rule_annotator import pre_annotate
    text = ("The city slept quietly through the winter months. "
            "Marie Curie discovered radium in 1898 after long effort. "
            "Later accounts describe the laboratory in detail.")
    ann = pre_annotate(text)
    assert ann["events"], "trigger 'discovered' must fire"
    for ev in ann["events"]:
        assert text[ev["start"]:ev["end"]] == ev["text"]
        assert ev["text"][0].isupper()
