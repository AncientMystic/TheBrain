import json
from pathlib import Path


def extract_ipynb(filepath: Path) -> dict:
    """Extract text from Jupyter Notebook (.ipynb)."""
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    text_parts = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "\n".join(source)
        else:
            source_text = source

        if cell_type == "markdown":
            text_parts.append(f"\n[MARKDOWN]\n{source_text}\n")
        elif cell_type == "code":
            text_parts.append(f"\n[CODE]\n{source_text}\n[/CODE]\n")
        else:
            text_parts.append(source_text)

    text = "\n".join(text_parts)

    metadata = {
        "title": filepath.stem,
        "author": "",
    }

    # Try to get title from first markdown heading
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = source
            for line in source_text.splitlines():
                if line.startswith("# "):
                    metadata["title"] = line[2:].strip()
                    break
            break

    return {
        "text": text,
        "metadata": metadata,
        "format": "ipynb",
    }