
import json
import torch
from pathlib import Path
import config
import logging
logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_model_dir = None
_missing_reported = False
_device = "cpu"

def _get_device():
    global _device
    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
            return _device
    except ImportError:
        logger.warning("Unexpected exception occurred", exc_info=True)
        pass
    try:
        import torch_directml
        _device = torch_directml.device()
        return _device
    except ImportError:
        logger.warning("Unexpected exception occurred", exc_info=True)
        pass
    _device = "cpu"
    return _device

def load_model():
    global _model, _tokenizer, _model_dir, _missing_reported, _json_start_token_id
    if _model is not None:
        return True
    model_dir = Path(config.DISTILLED_MODEL_DIR)
    if not model_dir.exists():
        if not _missing_reported:
            print("  (Distilled extractor model directory not found; will fallback to LLM)")
            _missing_reported = True
        return False
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if _tokenizer.pad_token_id is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
        _model.config.pad_token_id = _tokenizer.pad_token_id
        # Token ID for '{' to force JSON start
        _json_start_token_id = _tokenizer.convert_tokens_to_ids("{")
        if _json_start_token_id is None:
            _json_start_token_id = _tokenizer.encode("{", add_special_tokens=False)[0]
        device = _get_device()
        _model.to(device)
        _model.eval()
        _model_dir = model_dir
        print(f"  (Distilled extractor loaded on {device})")
        return True
    except Exception as e:
        print(f"  (Failed to load distilled extractor: {e})")
        _model = None
        logger.warning("Unexpected exception occurred", exc_info=True)
        return False

def generate_extraction(chunk_text):
    """Generate extraction JSON for a single chunk."""
    if not load_model():
        return None
    input_text = "extract: " + chunk_text[:1000]
    inputs = _tokenizer([input_text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=getattr(config, "DISTILLED_MAX_LENGTH", 1024),
            num_beams=1,
            do_sample=False,
            forced_decoder_ids=[(0, _json_start_token_id)],
        )
    raw = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        data = json.loads(raw)
        if any(data.get(k) for k in ["facts", "entities", "people", "locations"]):
            return data
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        pass
    return None

def generate_extractions_batch(texts, max_length=None, num_beams=None, batch_size=None):
    """Batch generation for speed. Returns list of parsed dicts or None."""
    if not load_model():
        return [None] * len(texts)
    if max_length is None:
        max_length = getattr(config, "DISTILLED_MAX_LENGTH", 512)
    if num_beams is None:
        num_beams = getattr(config, "DISTILLED_NUM_BEAMS", 1)
    if batch_size is None:
        batch_size = getattr(config, "DISTILLED_BATCH_SIZE", 2)

    results = [None] * len(texts)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        prefixed = ["extract: " + t[:1000] for t in batch_texts]
        inputs = _tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=1,
                do_sample=False,
                forced_decoder_ids=[(0, _json_start_token_id)],
            )
        for i, seq in enumerate(outputs):
            raw = _tokenizer.decode(seq, skip_special_tokens=True)
            try:
                data = json.loads(raw)
                if any(data.get(k) for k in ["facts", "entities", "people", "locations"]):
                    results[start + i] = data
            except Exception:
                logger.warning("Unexpected exception occurred", exc_info=True)
                pass
    return results

def is_available():
    return _model is not None
