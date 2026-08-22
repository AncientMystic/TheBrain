"""
Automatic download of ONNX NER model from Hugging Face.
Uses huggingface_hub to download snapshot.
"""
import os
import shutil
from pathlib import Path
import config

def download_onnx_model():
    """Download the ONNX model if not already present."""
    if not config.FAST_EXTRACTOR_ENABLED:
        return False
    model_dir = Path(config.FAST_EXTRACTOR_MODEL_DIR)
    if model_dir.exists() and any(model_dir.iterdir()):
        if config.DEBUG_VERBOSE:
                print(f"ONNX model already exists at {model_dir}")
        return True
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading ONNX model {config.FAST_EXTRACTOR_MODEL_NAME}...")
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=config.FAST_EXTRACTOR_MODEL_NAME, local_dir=model_dir)
        print("Download complete.")
        return True
    except ImportError:
        print("huggingface_hub not installed. Please install: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"Failed to download ONNX model: {e}")
        print("Falling back to rule-based extraction only.")
        return False
