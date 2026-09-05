# Model Card — TheBrain Extraction Gate

## Intended Use
Regime-conditioned extraction triage (which chunks need full LLM). Not a truth classifier.

## Training Data
- Source: `gate_training_data` (chunk spectral features + verified-fact labels)
- Splits: 5-fold CV, deterministic seeds, quantile knots documented in provenance ledger

## Regime Coverage
- Corners populated: (fill from `audit_regimes.py` marginal/joint rates)
- Sparsely represented: (list corners with <5% occupancy — do not trust there)

## Performance
- Validation BCE + prime/even support + anchor coherence (see tune log)
- Falsification: noise/small-n/adversarial/anti-prime suites must pass

## Limitations
- Single-topic docs: novelty floor + recall priority mitigate, still monitor stage counts
- Small models: use small-safe batches (2), solo retry; large batches overflow context
- Gate is triage only; truth decided by verification layers + review queue

## Provenance
- Run ID + root digest from `provenance_ledger`; replay via `verify_run.py`
- Tolerances: 1e-6 obj / 1e-4 params / <1% active mismatch (heterogeneous)
