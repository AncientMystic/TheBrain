from pathlib import Path


def extract_text(filepath: Path) -> dict:
    """Extract plain text from .txt/.text files."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    return {
        "text": text,
        "metadata": {
            "title": filepath.stem,
            "author": "",
        },
        "format": "text",
    }