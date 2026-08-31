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

    def _verify_single(self, fact: Dict, accepted_facts: List[Dict]) -> Dict:
        fact = fact.copy()
        triple = self._extract_triple(fact)
        fact.update(triple)

        layers = [self._symstep(fact, accepted_facts)]
        # Gated verification with identity default if untrained
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
                    from core.embeddings import get_embedding
                    emb = get_embedding(fact_text)
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

        # Adjust confidence based on trusted standards
        try:
            from core.embeddings import get_embedding
            from core.fact_normalizer import normalize_name
            fact_emb = get_embedding(fact.get("fact_text", ""))
            if fact_emb is not None:
                conn = db.db_connect("verification_standards")
                cur = conn.cursor()
                cur.execute("SELECT statement, negation, confidence FROM verified_standards WHERE truth_status IN ('admin_claim','verified_true')")
                standards = cur.fetchall()
                conn.close()
                if standards:
                    import numpy as np
                    fact_vec = np.array(fact_emb, dtype=np.float32)
                    fact_norm = np.linalg.norm(fact_vec)
                    best_sim = 0.0
                    best_neg = 0
                    best_conf = 1.0
                    for std in standards:
                        std_emb = get_embedding(std[0])
                        if std_emb is None:
                            continue
                        std_vec = np.array(std_emb, dtype=np.float32)
                        std_norm = np.linalg.norm(std_vec)
                        if std_norm == 0 or fact_norm == 0:
                            continue
                        sim = float(np.dot(fact_vec, std_vec) / (fact_norm * std_norm))
                        if sim > best_sim:
                            best_sim = sim
                            best_neg = std[1]
                            best_conf = std[2] if std[2] is not None else 1.0
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
        for fact in facts:
            vfact = self._verify_single(fact, self.accepted_facts)
            if vfact["verification_status"] in ("verified", "partially_verified"):
                self.accepted_facts.append(vfact)
            results.append(vfact)
        return results

    def verify_single_online(self, fact: Dict, accepted_facts: List[Dict] = None) -> Dict:
        if accepted_facts is None:
            accepted_facts = self.accepted_facts
        return self._verify_single(fact, accepted_facts)
