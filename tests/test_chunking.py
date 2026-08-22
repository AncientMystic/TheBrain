import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.text_utils import chunk_text

def test_chunking_basic():
    text = "Hello. This is a test. It should be split into chunks."
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1

def test_chunking_no_overlap():
    text = "This is a long text that will be split without overlap because overlap is zero."
    chunks = chunk_text(text, chunk_size=20, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 20  # approximate
