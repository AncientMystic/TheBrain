"""P1 pattern regression: new annotation buckets + the two fixed bugs."""
from extraction.rule_annotator import pre_annotate

DOC = """By Jane Smith
PARIS — Treaty text here.
MARIE: Hello world.
JOHN: Hi there.
Published 2020-01-15, see p. 42 and pp. 44-47.
ISBN 978-0-123456-78-9, ISSN 1234-567X, doi 10.1000/xyz123.
As shown in Curie (1903), March 2020 was key, c. 1898 too.
Chapter 3
"""


def _texts(bucket):
    return [x["text"] for x in pre_annotate(DOC).get(bucket, [])]


def test_byline_case_insensitive():
    assert any("Jane Smith" in b for b in _texts("bylines"))


def test_dateline_skips_speakers():
    assert _texts("datelines") == ["PARIS"]


def test_speakers():
    assert sorted(_texts("speakers")) == ["JOHN", "MARIE"]


def test_pages_and_identifiers():
    pages = _texts("pages")
    assert "p. 42" in pages and "pp. 44-47" in pages
    ids = _texts("identifiers")
    assert any("978-0-123456-78-9" in i for i in ids)
    assert any("1234-567X" in i for i in ids)
    assert any("10.1000/xyz123" in i for i in ids)


def test_citations_chapters_dates():
    assert any("Curie (1903)" in x for x in _texts("citations"))
    assert any(x.strip() == "Chapter 3" for x in _texts("chapters"))
    dates = _texts("dates")
    assert "2020-01-15" in dates and "March 2020" in dates
    assert any("1898" in d for d in dates)


def test_existing_buckets_untouched():
    a = pre_annotate("Marie Curie was born in Warsaw, Poland in 1867.")
    assert [x["text"] for x in a["years"]] == ["1867"]
    assert any(x["text"] == "Marie Curie" for x in a["people"])
    assert "Poland" in [x["text"] for x in a["locations"]]
