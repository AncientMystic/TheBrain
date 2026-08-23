"""
ONNX NER inference using onnxruntime.
Loads model from config.FAST_EXTRACTOR_MODEL_DIR.
"""
import numpy as np
import config
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class OnnxNERExtractor:
    def __init__(self):
        self.session = None
        self.tokenizer = None
        self.id2label = None
        if ort and config.FAST_EXTRACTOR_ENABLED:
            self._load_model()

    def _load_model(self):
        model_dir = Path(config.FAST_EXTRACTOR_MODEL_DIR)
        if not model_dir.exists():
            print("ONNX model directory not found.")
            return
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            print("No .onnx file found in model directory.")
            return
        try:
            self.session = ort.InferenceSession(str(onnx_files[0]))
            # For tokenizer, try to use transformers if available
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                # Try to load id2label from model config; fallback to default BERT-NER mapping
                self.id2label = None
                config_path = model_dir / "config.json"
                if config_path.exists():
                    try:
                        import json
                        with open(config_path, "r", encoding="utf-8") as cf:
                            model_cfg = json.load(cf)
                        id2label_raw = model_cfg.get("id2label")
                        if id2label_raw:
                            self.id2label = {int(k): v for k, v in id2label_raw.items()}
                    except Exception:
                        pass
                if not self.id2label:
                    self.id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
                                     5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
            except ImportError:
                print("transformers not installed, cannot load tokenizer.")
                self.session = None
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            self.session = None

    def extract_entities(self, text):
        """Return list of (entity_type, entity_text, confidence)."""
        if not self.session or not self.tokenizer:
            return []
        try:
            inputs = self.tokenizer(text, return_tensors="np", truncation=True, padding=True)
            ort_inputs = {name: inputs[name] for name in inputs}
            outputs = self.session.run(None, ort_inputs)
            # Assuming first output is logits [batch, seq_len, num_labels]
            logits = outputs[0][0]  # first batch
            preds = np.argmax(logits, axis=-1)
            probs = np.max(np.exp(logits)/np.sum(np.exp(logits), axis=-1, keepdims=True), axis=-1)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            entities = []
            current_entity = []
            current_type = None
            current_conf = []
            for token, pred, prob in zip(tokens, preds, probs):
                label = self.id2label.get(pred, "O")
                if label.startswith("B-"):
                    if current_entity:
                        entities.append(("".join(current_entity).replace("##", ""), current_type, float(np.mean(current_conf))))
                    current_entity = [token]
                    current_type = label[2:]
                    current_conf = [prob]
                elif label.startswith("I-") and current_entity:
                    current_entity.append(token)
                    current_conf.append(prob)
                else:
                    if current_entity:
                        entities.append(("".join(current_entity).replace("##", ""), current_type, float(np.mean(current_conf))))
                        current_entity = []
                        current_type = None
                        current_conf = []
            if current_entity:
                entities.append(("".join(current_entity).replace("##", ""), current_type, float(np.mean(current_conf))))
            return entities
        except Exception as e:
            print(f"ONNX inference error: {e}")
            return []
