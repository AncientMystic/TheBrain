"""Offline benchmark harness: ranker recall@k/MRR + verifier precision/recall.

Deterministic synthetic fixtures, no backends, no database. Complements
scripts/evaluate.py (live-DB counterpart) with CI-runnable numbers that pin
current behavior — regressions here mean ranking or logic actually changed.
"""
from retrieval.ranking import FallbackRanker
from reasoning.verify import verify_symstep


def _dp(i, text, dtype="fact", conf=0.5):
    return {"id": f"fact:{i}", "type": dtype, "text": text, "confidence": conf}


def _recall_mrr(ranked_ids, relevant, k=5):
    top = ranked_ids[:k]
    hits = [i for i in top if i in relevant]
    recall = len(set(hits)) / len(relevant)
    mrr = 0.0
    for rank, fid in enumerate(top, 1):
        if fid in relevant:
            mrr = 1.0 / rank
            break
    return recall, mrr


def test_ranker_recall_and_mrr():
    ranker = FallbackRanker()
    query = "What did Marie Curie discover?"
    corpus = [_dp(1, "Marie Curie discovered radium and polonium.", conf=0.9),
              _dp(2, "Radium glows faintly blue in the dark.", conf=0.4),
              _dp(3, "The weather in Paris was mild that year.", conf=0.4),
              _dp(4, "Pierre Curie married Marie Sklodowska.", conf=0.6),
              _dp(5, "Polonium was named after Poland.", conf=0.5)]
    scored = sorted(((ranker.score(query, dp, [], None), dp["id"]) for dp in corpus), reverse=True)
    recall, mrr = _recall_mrr([fid for _, fid in scored], {"fact:1"}, k=5)
    assert recall == 1.0
    assert mrr == 1.0  # exact-overlap fact must rank first
    print(f"\nranker recall@5={recall} mrr={mrr}")


def _claim(s, p, o):
    return {"subject": s, "predicate": p, "object": o}


def test_verifier_precision_recall():
    priors = [_claim("Alice", "pet", "Cat")]
    positives = [_claim("Alice", "pet", "Cat")]
    negatives = [_claim("Alice", "pet", "Dog")]
    tp = sum(1 for c in positives if verify_symstep(c, priors))
    fp = sum(1 for c in negatives if verify_symstep(c, priors))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, len(positives))
    assert precision == 1.0, "contradiction passed as consistent"
    assert recall == 1.0, "entailment rejected"
    print(f"\nverifier precision={precision} recall={recall}")
