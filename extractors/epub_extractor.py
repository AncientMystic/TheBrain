from pathlib import Path
from bs4 import BeautifulSoup

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None


def extract_epub(filepath: Path) -> dict:
    """Extract text from .epub using ebooklib."""
    if ebooklib is None:
        raise ImportError("ebooklib is required for .epub files. Install with: pip install ebooklib")

    book = epub.read_epub(str(filepath))
    text_parts = []
    title = filepath.stem
    author = ""

    # Get metadata
    try:
        if book.get_metadata('DC', 'title'):
            title = book.get_metadata('DC', 'title')[0][0]
        if book.get_metadata('DC', 'creator'):
            author = book.get_metadata('DC', 'creator')[0][0]
    except Exception:
        pass

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        try:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            # Remove scripts/styles
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if text:
                text_parts.append(text)
        except Exception:
            continue

    full_text = "\n\n".join(text_parts)

    return {
        "text": full_text,
        "metadata": {
            "title": title,
            "author": author,
        },
        "format": "epub",
    }