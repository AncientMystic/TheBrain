"""Reranker ONNX-first: loads without torch/CUDA, ranks relevant first."""
from core.reranker import get_reranker


def test_onnx_backend_ranks():
    r = get_reranker()
    assert r.available, "reranker failed to load on any backend"
    assert r.backend == "onnx", f"expected onnx-first, got {r.backend}"
    q = "Zealandia continent discovery"
    texts = [
        "Bananas are yellow fruit grown in tropics.",
        "Zealandia spans 4.9 million square km beneath New Zealand.",
        "Quantum entanglement puzzled Einstein for years.",
    ]
    scores = r.score(q, texts)
    assert len(scores) == 3 and all(0.0 <= s <= 1.0 for s in scores)
    assert scores.index(max(scores)) == 1, f"relevant passage must win: {scores}"
