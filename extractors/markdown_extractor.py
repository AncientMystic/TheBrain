from pathlib import Path
import re
import logging
logger = logging.getLogger(__name__)


def extract_markdown(filepath: Path) -> dict:
    """
    Extract text and metadata from Markdown file.
    Extracts YAML frontmatter if present.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    metadata = {}
    text = raw

    # YAML frontmatter
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end != -1:
            frontmatter = raw[3:end].strip()
            text = raw[end+3:].strip()
            # Parse simple key: value pairs
            for line in frontmatter.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key:
                        metadata[key] = val

    # Extract headings, code blocks etc.
    text_parts = []
    in_code = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            if in_code:
                text_parts.append("\n[CODE]\n")
            else:
                text_parts.append("\n[/CODE]\n")
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heading = line.lstrip("#").strip()
            text_parts.append(f"\n[H{level}] {heading}\n")
        else:
            text_parts.append(line)
    text = "\n".join(text_parts)

    title = metadata.get("title", filepath.stem)
    author = metadata.get("author", "")

    return {
        "text": text,
        "metadata": {
            "title": title,
            "author": author,
            "year": metadata.get("year", ""),
            "tags": metadata.get("tags", []),
        },
        "format": "markdown",
    }
