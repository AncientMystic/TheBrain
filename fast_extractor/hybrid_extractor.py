"""
Hybrid extractor combining rules, ONNX NER, and optionally small LLM.
Returns structured items with confidence, identifying low-confidence items for LLM verification.
"""
import config
from fast_extractor.model_download import download_onnx_model
from fast_extractor.onnx_ner import OnnxNERExtractor
from fast_extractor.rule_extractor import extract_entities_rules

class FastExtractor:
    def __init__(self):
        self.onnx_extractor = None
        if config.FAST_EXTRACTOR_ENABLED:
            download_onnx_model()
            self.onnx_extractor = OnnxNERExtractor()
        self.entities = []
        self.dates = []
        self.locations = []
        self.people = []
        self.organizations = []

    def extract(self, text):
        """Extract structured entities from text. Returns dict of lists with confidence."""
        entities = []
        # Rule-based
        rule_entities = extract_entities_rules(text)
        for ent_type, ent_text, conf in rule_entities:
            entities.append({"type": ent_type, "text": ent_text, "confidence": conf, "source": "rule"})
        # ONNX NER
        if self.onnx_extractor:
            onnx_entities = self.onnx_extractor.extract_entities(text)
            for ent_text, ent_type, conf in onnx_entities:
                # Map types
                if ent_type in ("PER", "PERSON"):
                    mapped_type = "PERSON"
                elif ent_type in ("ORG", "ORGANIZATION"):
                    mapped_type = "ORG"
                elif ent_type in ("LOC", "LOCATION", "GPE"):
                    mapped_type = "LOC"
                else:
                    mapped_type = "MISC"
                entities.append({"type": mapped_type, "text": ent_text, "confidence": conf, "source": "onnx"})

        # Deduplicate and merge
        merged = {}
        for ent in entities:
            key = (ent["type"], ent["text"].lower())
            if key not in merged or ent["confidence"] > merged[key]["confidence"]:
                merged[key] = ent
        final_entities = list(merged.values())

        # Separate into categories
        self.people = [e for e in final_entities if e["type"] == "PERSON"]
        self.organizations = [e for e in final_entities if e["type"] == "ORG"]
        self.locations = [e for e in final_entities if e["type"] == "LOC"]
        self.dates = [e for e in final_entities if e["type"] == "DATE"]
        self.entities = final_entities

        return {
            "entities": final_entities,
            "people": self.people,
            "locations": self.locations,
            "dates": self.dates,
            "organizations": self.organizations,
        }

    def get_low_confidence_items(self, threshold=None):
        """Return items with confidence below threshold for LLM verification."""
        if threshold is None:
            threshold = config.FAST_EXTRACTOR_CONFIDENCE_THRESHOLD
        low = []
        for ent in self.entities:
            if ent["confidence"] < threshold:
                low.append(ent)
        return low
