import os
from pathlib import Path

import config
from core.file_utils import is_ignored_dir, is_ignored_file, is_text_file, normalize_path


def scan_files(input_path: str | Path, follow_symlinks: bool = False) -> list[Path]:
    """
    Recursively scan a file or folder and return list of text files to process.

    Args:
        input_path: file or directory path.
        follow_symlinks: whether to follow symbolic links.

    Returns:
        List of Path objects for text files.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        if is_text_file(input_path):
            return [input_path]
        else:
            return []

    files = []
    for root, dirs, filenames in os.walk(input_path, followlinks=follow_symlinks):
        # Prune ignored directories
        dirs[:] = [
            d for d in dirs
            if not is_ignored_dir(d)
            and not d.startswith('.')
        ]
        for filename in filenames:
            if is_ignored_file(filename):
                continue
            filepath = Path(root) / filename
            if is_text_file(filepath):
                files.append(filepath)

    return files