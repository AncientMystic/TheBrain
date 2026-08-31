import hashlib
import mimetypes
import os
from pathlib import Path

import config


def get_file_hash(filepath: str | Path) -> str:
    """Return SHA1 hash of file content (same as Vision Organizer Deep)."""
    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")
    hasher = hashlib.sha1()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_extension(filepath: str | Path) -> str:
    """Return lowercase file extension without dot."""
    return Path(filepath).suffix.lower()


def is_text_file(filepath: str | Path) -> bool:
    """
    Determine if a file should be processed as text.
    Uses extension allowlist and MIME fallback.
    """
    ext = get_file_extension(filepath)
    if ext in config.TEXT_EXTENSIONS:
        return True

    mime, _ = mimetypes.guess_type(str(filepath))
    if mime and mime.startswith(("text/", "application/json", "application/xml")):
        return True

    return False


def read_file_binary(filepath: str | Path) -> bytes:
    """Read entire file as bytes."""
    with open(filepath, "rb") as f:
        return f.read()


def normalize_path(filepath: str | Path) -> str:
    """Return absolute normalized path."""
    return str(Path(filepath).resolve())


def is_ignored_dir(dirname: str) -> bool:
    """Check if directory should be skipped during scanning."""
    return dirname in config.IGNORE_DIRS


def is_ignored_file(filename: str) -> bool:
    """Check if file should be skipped."""
    return filename in config.IGNORE_FILES