
"""
Verification Manager for TheBrain.

Orchestrates multi-layer verification of extracted facts:
- SymStep: constraint propagation against previously accepted facts.
- VeriCoT: logical verification using formal triples.
- R-CoT: reverse chain-of-thought verification.
- ARES: probabilistic entailment using NLI or heuristic.
"""

import logging
from typing import List, Dict, Optional

import config
from core import db
from reasoning.verify import (
    verify_symstep,
    verify_vericot,
    verify_rcot,
    extract_triple_from_text,
)

logger = logging.getLogger(__name__)


class VerificationManager:
    """
    Handles verification of facts with adaptive escalation.
    Maintains an internal state of accepted facts for SymStep checks.
    """

    def __init__(self, kg=None, use_nli: bool = False):
        self.accepted_facts: List[Dict] = []
        self.kg = kg
        self.use_nli = use_nli
        self.nli_pipeline = None
        self.standards_cache = None
        if self.use_nli:
            try:
                from transformers import pipeline
                self.nli_pipeline = pipeline("text-classification", model="roberta-large-mnli")
            except Exception as e:
                logger.warning(f"Failed to load NLI model: {e}. Using heuristic ARES.")

    def _extract_triple(self, fact: Dict) -> Dict:
        """Extract subject, predicate, object from fact if possible."""
        if all(k in fact for k in ("subject", "predicate", "object")):
            def _coerce(v):
                return str(v) if not isinstance(v, str) else v
            return {
                "subject": _coerce(fact.get("subject")) or "",
                "predicate": _coerce(fact.get("predicate")) or "",
                "object": _coerce(fact.get("object")) or "",
                "negation": fact.get("negation", 0) or 0,
            }
        # Use cached triple if available
        cached = fact.get("_cached_triple")
        if cached and isinstance(cached, dict):
            triple = cached.copy()
            triple["negation"] = fact.get("negation", 0)
            return triple
        triple = extract_triple_from_text(fact.get("fact_text", ""))
        if triple:
            triple["negation"] = fact.get("negation", 0)
            return triple
        def _coerce(v):
            return str(v) if not isinstance(v, str) else v
        return {
            "subject": (_coerce(fact.get("canonical_value")) or _coerce(fact.get("fact_text")) or "")[:50],
            "predicate": "has_fact",
            "object": _coerce(fact.get("fact_text")) or "",
            "negation": fact.get("negation", 0) or 0,
        }

    def _symstep(self, fact: Dict, prior_facts: List[Dict]) -> Dict:
        triple = self._extract_triple(fact)
        claim = fact.copy()
        claim.update(triple)
        claim['_prior_claims'] = prior_facts
        ok = verify_symstep(claim, prior_facts)
        return {"layer": "symstep", "verified": ok, "confidence": 1.0 if ok else 0.0}

    def _vericot(self, fact: Dict) -> Dict:
        ok = verify_vericot(fact.get("fact_text", ""), "", self.kg)
        return {"layer": "vericot", "verified": ok, "confidence": 0.8 if ok else 0.0}

    def _rcot(self, fact: Dict) -> Dict:
        conclusion = fact.get("object") or fact.get("canonical_value") or fact.get("fact_text", "")
        ok = verify_rcot(conclusion, self.kg)
        return {"layer": "rcot", "verified": ok, "confidence": 0.7 if ok else 0.0}

    def _ares(self, fact: Dict, accepted_facts: List[Dict]) -> Dict:
        if self.nli_pipeline and accepted_facts:
            premise = " ".join([f.get("fact_text", "") for f in accepted_facts[-3:]])
            hypothesis = fact.get("fact_text", "")
            try:
                result = self.nli_pipeline({"text": premise, "text_pair": hypothesis})
                label = result[0]["label"]
                score = result[0]["score"]
                if label == "entailment":
                    confidence = score
                elif label == "contradiction":
                    confidence = 0.0
                else:
                    confidence = score * 0.5
                return {"layer": "ares", "verified": confidence > 0.5, "confidence": confidence}
            except Exception as e:
                logger.warning(f"NLI failed: {e}. Falling back to heuristic.")
        sym = self._symstep(fact, accepted_facts)
        return {"layer": "ares", "verified": sym["verified"], "confidence": 0.5 if sym["verified"] else 0.0}

    def _load_standards(self):
        """Load and cache standards embeddings once."""
        if self.standards_cache is not None:
            return self.standards_cache
        conn = db.db_connect("verification_standards")
        cur = conn.cursor()
        cur.execute("SELECT statement, negation, confidence FROM verified_standards WHERE truth_status IN ('admin_claim','verified_true')")
        rows = cur.fetchall()
        conn.close()
        if not rows:
            self.standards_cache = []
            return self.standards_cache

        statements = [r["statement"] for r in rows]
        from core.embeddings import get_embeddings_batch
        embs = get_embeddings_batch(statements, space='hyperbolic', model=config.EMBEDDING_MODEL)
        self.standards_cache = []
        for r, emb in zip(rows, embs):
            if emb is not None:
                self.standards_cache.append({
                    "statement": r["statement"],
                    "negation": r["negation"],
                    "confidence": r["confidence"] or 1.0,
                    "embedding": emb,
                })
        return self.standards_cache

    def _batch_embed_facts(self, facts):
        """Precompute embeddings for all facts to avoid repeated HTTP calls."""
        from core.embeddings import get_embeddings_batch
        texts = [f.get("fact_text", "") for f in facts]
        embs = get_embeddings_batch(texts, space='hyperbolic', model=config.EMBEDDING_MODEL)
        return {f.get("fact_text", ""): emb for f, emb in zip(facts, embs) if emb is not None}

    def _batch_extract_triples(self, facts):
        """Extract triples using regex-only locally, then one LLM batch for failures."""
        import re
        from core.llm import call_model_json

        # Local regex patterns (same as extract_triple_from_text without LLM fallback)
        patterns = [
            (r"(.+?)\s+is\s+(?:a|an)\s+(.+)", "is_a"),
            (r"(.+?)\s+are\s+(?:a|an|the)?\s*(.+)", "is_a"),
            (r"(.+?)\s+has\s+(.+)", "has"),
            (r"(.+?)\s+have\s+(.+)", "has"),
            (r"(.+?)\s+located\s+in\s+(.+)", "located_in"),
            (r"(.+?)\s+is\s+located\s+in\s+(.+)", "located_in"),
            (r"(.+?)\s+works\s+for\s+(.+)", "works_for"),
            (r"(.+?)\s+is\s+part\s+of\s+(.+)", "part_of"),
            (r"(.+?)\s+belongs\s+to\s+(.+)", "belongs_to"),
            (r"(.+?)\s+caused\s+(.+)", "caused_by"),
            (r"(.+?)\s+produced\s+(.+)", "produced"),
            (r"(.+?)\s+discovered\s+(.+)", "discovered"),
        ]

        triple_cache = {}
        fallback_texts = []

        for fact in facts:
            text = fact.get("fact_text", "")
            if not text:
                triple_cache[text] = None
                continue

            found = False
            for pattern, pred in patterns:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    subject = m.group(1).strip()
                    obj = m.group(2).strip()
                    subject = re.sub(r'^(a|an|the)\s+', '', subject, flags=re.IGNORECASE)
                    obj = re.sub(r'^(a|an|the)\s+', '', obj, flags=re.IGNORECASE)
                    triple_cache[text] = {"subject": subject, "predicate": pred, "object": obj}
                    found = True
                    break
            if not found:
                fallback_texts.append(text)

        if fallback_texts:
            prompt = (
                "Extract subject, predicate, and object from each sentence as JSON objects with keys "
                "'subject', 'predicate', 'object'. Return a JSON array of objects, one per sentence, in the same order.\n\n"
                "Sentences:\n"
            )
            for i, t in enumerate(fallback_texts):
                prompt += f"{i+1}. {t}\n"
            prompt += "\nReturn only JSON array."

            try:
                data = call_model_json(prompt, max_tokens=2048, unwrap_list=False)
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        if isinstance(item, dict) and all(k in item for k in ("subject", "predicate", "object")):
                            if idx < len(fallback_texts):
                                triple_cache[fallback_texts[idx]] = {
                                    "subject": str(item.get("subject", "")),
                                    "predicate": str(item.get("predicate", "")),
                                    "object": str(item.get("object", "")),
                                }
                elif isinstance(data, dict) and len(fallback_texts) == 1:
                    if all(k in data for k in ("subject", "predicate", "object")):
                        triple_cache[fallback_texts[0]] = {
                            "subject": str(data.get("subject", "")),
                            "predicate": str(data.get("predicate", "")),
                            "object": str(data.get("object", "")),
                        }
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"Batch triple extraction error: {e}")

        for t in fallback_texts:
            if t not in triple_cache:
                triple_cache[t] = None
        return triple_cache

    def _verify_single(self, fact: Dict, accepted_facts: List[Dict]) -> Dict:
        fact = fact.copy()
        triple = self._extract_triple(fact)
        fact.update(triple)

        layers = [self._symstep(fact, accepted_facts)]
        if getattr(config, "USE_GATED_VERIFICATION", False):
            try:
                from pathlib import Path
                from reasoning.verification_gate import VerificationGate
                gate_path = Path(config.BASE_DIR) / "models" / "verification_gate.json"
                if gate_path.exists():
                    import numpy as np
                    from reasoning.verification_gate import VerificationGate
                    gate = VerificationGate()
                    gate.load(gate_path)
                    from core.spectral import compute_spectral_features
                    fact_text = fact.get("fact_text", "")
                    emb = fact.get("_fact_embedding")
                    if emb is None:
                        from core.embeddings import get_embedding
                        emb = get_embedding(fact_text, model=config.EMBEDDING_MODEL)
                    if emb is not None:
                        features = compute_spectral_features(np.array([emb], dtype=np.float32))
                        weights = gate.forward(features)
                        for v in layers:
                            v["confidence"] *= weights.get(v["layer"], 1.0)
                else:
                    if config.DEBUG_VERBOSE:
                        print("    (Verification gate not trained, using unscaled confidences)")
            except Exception as e:
                if config.DEBUG_VERBOSE:
                    print(f"    (Gated verification error: {e})")

        conf = sum(v["confidence"] for v in layers if v["verified"]) / max(1, len(layers))
        if conf < getattr(config, "VERIFICATION_ESCALATION_THRESHOLD", 0.7):
            layers.append(self._vericot(fact))
            layers.append(self._rcot(fact))
            layers.append(self._ares(fact, accepted_facts))

        verified = any(v["verified"] for v in layers)
        verified_layers = [v for v in layers if v["verified"]]
        final_conf = sum(v["confidence"] for v in verified_layers) / len(verified_layers) if verified_layers else 0.0

        if verified and final_conf >= 0.7:
            status = "verified"
        elif verified:
            status = "partially_verified"
        else:
            status = "unverified"

        fact["verification_status"] = status
        fact["verification_layers"] = layers
        fact["confidence_final"] = final_conf
        fact["verified_by"] = "VerificationManager"

        # Adjust confidence based on cached trusted standards
        try:
            standards = self._load_standards()
            if standards and fact.get("fact_text"):
                fact_emb = fact.get("_fact_embedding")
                if fact_emb is None:
                    from core.embeddings import get_embedding
                    fact_emb = get_embedding(fact["fact_text"], model=config.EMBEDDING_MODEL)
                if fact_emb is not None:
                    import numpy as np
                    fact_vec = np.array(fact_emb, dtype=np.float32)
                    fact_norm = np.linalg.norm(fact_vec)
                    best_sim = 0.0
                    best_neg = 0
                    best_conf = 1.0
                    for std in standards:
                        std_vec = np.array(std["embedding"], dtype=np.float32)
                        std_norm = np.linalg.norm(std_vec)
                        if std_norm == 0 or fact_norm == 0:
                            continue
                        sim = float(np.dot(fact_vec, std_vec) / (fact_norm * std_norm))
                        if sim > best_sim:
                            best_sim = sim
                            best_neg = std["negation"]
                            best_conf = std["confidence"]
                    if best_sim > 0.9:
                        if best_neg == (fact.get("negation", 0) or 0):
                            fact["confidence_final"] = max(final_conf, 0.9)
                            fact["verification_status"] = "verified"
                            fact["verified_by"] = "standards_match"
                        else:
                            fact["confidence_final"] = min(final_conf, 0.3)
                            fact["verification_status"] = "disputed"
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"    (Standards calibration error: {e})")

        return fact

    def verify_batch(self, facts: List[Dict]) -> List[Dict]:
        self.accepted_facts = []
        results = []
        # Precompute fact embeddings once
        fact_embeddings = self._batch_embed_facts(facts)
        # Batch extract triples once
        triple_cache = self._batch_extract_triples(facts)
        for fact in facts:
            fact_text = fact.get("fact_text", "")
            if fact_text in fact_embeddings:
                fact["_fact_embedding"] = fact_embeddings[fact_text]
            if fact_text in triple_cache:
                fact["_cached_triple"] = triple_cache[fact_text]
            vfact = self._verify_single(fact, self.accepted_facts)
            vfact.pop("_fact_embedding", None)
            vfact.pop("_cached_triple", None)
            if vfact["verification_status"] in ("verified", "partially_verified"):
                self.accepted_facts.append(vfact)
            results.append(vfact)
        return results

    def verify_single_online(self, fact: Dict, accepted_facts: List[Dict] = None) -> Dict:
        if accepted_facts is None:
            accepted_facts = self.accepted_facts
        return self._verify_single(fact, accepted_facts)
