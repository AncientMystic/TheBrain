"""
Automatic download of ONNX NER model from Hugging Face.
Uses huggingface_hub to download snapshot.
"""
import os
import shutil
from pathlib import Path
import config
import logging
logger = logging.getLogger(__name__)

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
        logger.warning("Unexpected exception occurred", exc_info=True)
        return False
    except Exception as e:
        print(f"Failed to download ONNX model: {e}")
        print("Falling back to rule-based extraction only.")
        logger.warning("Unexpected exception occurred", exc_info=True)
        return False


def download_gliner_model():
    """Download the GLiNER ONNX model if not already present (opt-in path)."""
    if not getattr(config, "GLINER_ENABLED", False):
        return False
    model_dir = Path(getattr(config, "GLINER_MODEL_DIR",
                             str(Path(config.BASE_DIR) / "models" / "gliner_ner")))
    if (model_dir / "onnx").exists() or (model_dir / "model_quantized.onnx").exists():
        if config.DEBUG_VERBOSE:
            print(f"GLiNER model already exists at {model_dir}")
        return True
    try:
        from huggingface_hub import snapshot_download
        repo = getattr(config, "GLINER_MODEL_REPO", "onnx-community/gliner_small-v2.1")
        print(f"Downloading GLiNER model {repo}...")
        model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, local_dir=model_dir)
        print("GLiNER download complete.")
        return True
    except ImportError:
        print("huggingface_hub not installed. Please install: pip install huggingface_hub")
        logger.warning("Unexpected exception occurred", exc_info=True)
        return False
    except Exception as e:
        print(f"Failed to download GLiNER model: {e}")
        print("Falling back to bert NER path.")
        logger.warning("Unexpected exception occurred", exc_info=True)
        return False
