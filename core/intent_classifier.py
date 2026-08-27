"""
Intent classifier using a zero-shot model.
"""
import config
from pathlib import Path

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class IntentClassifier:
    def __init__(self):
        self.classifier = None
        self.available = False
        if pipeline and getattr(config, "INTENT_CLASSIFIER_ENABLED", True):
            self._load_model()

    def _load_model(self):
        model_dir = Path(config.INTENT_CLASSIFIER_MODEL_DIR)
        if not model_dir.exists():
            print("Intent classifier model directory not found.")
            return
        try:
            self.classifier = pipeline("zero-shot-classification", model=str(model_dir))
            self.available = True
        except Exception as e:
            print(f"Failed to load intent classifier: {e}")

    def classify(self, query, candidate_labels):
        """Return most likely label."""
        if not self.available or not query:
            return "general"
        try:
            result = self.classifier(query, candidate_labels)
            return result["labels"][0]
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"Intent classifier error: {e}")
            return "general"


_intent_classifier = None

def get_intent_classifier():
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
