"""Collective linking coherence."""
import numpy as np
from core.entity_linking import collective_link


def test_greedy_and_coherence():
    cands = {
        "apple": [("Apple Inc.", np.ones(8, dtype=np.float32) * 0.1), ("apple fruit", np.ones(8, dtype=np.float32) * 0.9)],
        "iphone": [("iPhone", np.ones(8, dtype=np.float32) * 0.1), ("iron phone", np.ones(8, dtype=np.float32) * 0.9)],
    }
    res = collective_link(["apple", "iphone"], lambda m: cands[m])
    assert res[0] == "Apple Inc."
    assert res[1] == "iPhone"


def test_empty():
    assert collective_link([], lambda m: []) == {}
