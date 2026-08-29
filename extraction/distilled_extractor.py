
import json
import numpy as np
from pathlib import Path
import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_model = None
_tokenizer = None
_model_dir = None
_missing_reported = False

def load_model():
    global _model, _tokenizer, _model_dir
    if _model is not None:
        return True
    global _missing_reported
    model_dir = Path(config.DISTILLED_MODEL_DIR)
    if not model_dir.exists():
        if not _missing_reported:
            print("  (Distilled extractor model directory not found; will fallback to LLM)")
            _missing_reported = True
        return False
    try:
        _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        _model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
        _model.eval()
        _model_dir = model_dir
        print("  (Distilled extractor loaded)")
        return True
    except Exception as e:
        print(f"  (Failed to load distilled extractor: {e})")
        _model = None
        return False

def generate_extraction(chunk_text):
    """Return parsed dict from distilled model or None if invalid."""
    if not load_model():
        return None
    input_text = "extract: " + chunk_text[:1000]
    inputs = _tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = _model.generate(
        **inputs,
        max_length=1024,
        num_beams=4,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    # Get average log-prob of generated tokens
    scores = outputs.scores
    if scores:
        probs = [s.softmax(dim=-1).max().item() for s in scores]
        avg_conf = sum(probs) / len(probs)
    else:
        avg_conf = 0.0
    if avg_conf < getattr(config, "DISTILLED_CONFIDENCE_THRESHOLD", 0.5):
        return None
    raw = _tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    try:
        data = json.loads(raw)
        # Basic validation: must have at least one non-empty key
        if any(data.get(k) for k in ["facts", "entities", "people", "locations"]):
            return data
    except Exception:
        pass
    return None

def is_available():
    return _model is not None
