from pathlib import Path

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None


def extract_rtf(filepath: Path) -> dict:
    """Extract text from .rtf using striprtf."""
    if rtf_to_text is None:
        raise ImportError("striprtf is required for .rtf files. Install with: pip install striprtf")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        rtf_content = f.read()
    text = rtf_to_text(rtf_content)

    return {
        "text": text,
        "metadata": {
            "title": filepath.stem,
            "author": "",
        },
        "format": "rtf",
    }