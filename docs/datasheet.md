# Datasheet — Gate Training Data

## Motivation
Triage labels (did chunk yield verified facts?) for gate tuning. Collected during guided-learning.

## Composition
- Rows: `(chunk_hash, features[44], label)` — features are spectral aggregates, not raw text
- Label: 1 if chunk produced ≥1 verified/partially_verified fact, else 0
- No PII beyond chunk hashes; raw text stays in `document_chunks`

## Collection
- Automatic on ingestion when `USE_PRIME_EVEN_GATE` enabled; `tune_gate_weights.py` consumes
- Splits: deterministic 5-fold by index mod 5; seeds recorded in ledger

## Preprocessing
- Features standardized per spectral block; quantile knots from training quantiles (record vectors)
- Cleaning: drop NaN/inf rows, dedup by chunk_hash

## Uses / Distribution / Maintenance
- Uses: gate CV only, not for LLM prompting or verification thresholds
- Distribution: local SQLite `key_facts.gate_training_data`, hashed in ledger
- Maintenance: append on new ingestion, re-tune quarterly or when precision drifts; see falsification suites
