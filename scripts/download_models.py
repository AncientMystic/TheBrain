#!/usr/bin/env python3
"""
Auto-download neural models for TheBrain.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def download_model(repo_id, local_dir, enabled_flag):
    if not enabled_flag:
        print(f"[SKIP] {repo_id}: disabled by config")
        return True

    local_path = Path(local_dir)
    if local_path.exists() and any(local_path.iterdir()):
        print(f"[SKIP] {repo_id}: already exists")
        return True

    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo_id} to {local_path}...")
        local_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=local_path)
        print(f"Downloaded {repo_id}.")
        return True
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")
        return False


def main():
    print("Starting neural model downloads...\n")
    success = True

    if getattr(config, "RERANKER_ENABLED", True):
        success &= download_model(config.RERANKER_MODEL_REPO, config.RERANKER_MODEL_DIR, True)
    else:
        print("[SKIP] Reranker disabled")

    if getattr(config, "LOCAL_EMBEDDER_ENABLED", True):
        success &= download_model(config.LOCAL_EMBEDDER_MODEL_REPO, config.LOCAL_EMBEDDER_MODEL_DIR, True)
    else:
        print("[SKIP] Local embedder disabled")

    if getattr(config, "INTENT_CLASSIFIER_ENABLED", True):
        success &= download_model(config.INTENT_CLASSIFIER_MODEL_REPO, config.INTENT_CLASSIFIER_MODEL_DIR, True)
    else:
        print("[SKIP] Intent classifier disabled")

    if not success:
        print("\nSome downloads failed. TheBrain will fall back to existing methods.")
    else:
        print("\nAll downloads completed successfully.")


if __name__ == "__main__":
    main()
