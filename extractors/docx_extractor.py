from pathlib import Path
import logging
logger = logging.getLogger(__name__)

try:
    import docx
except ImportError:
    docx = None


def extract_docx(filepath: Path) -> dict:
    """Extract text from .docx using python-docx."""
    if docx is None:
        raise ImportError("python-docx is required for .docx files. Install with: pip install python-docx")

    document = docx.Document(str(filepath))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)

    # Extract metadata
    metadata = {
        "title": filepath.stem,
        "author": "",
    }
    try:
        core_props = document.core_properties
        if core_props.title:
            metadata["title"] = core_props.title
        if core_props.author:
            metadata["author"] = core_props.author
    except Exception:
        logger.warning("Unexpected exception occurred", exc_info=True)
        pass

    return {
        "text": text,
        "metadata": metadata,
        "format": "docx",
    }
