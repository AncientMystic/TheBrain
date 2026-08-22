from pathlib import Path
from bs4 import BeautifulSoup


def extract_html(filepath: Path) -> dict:
    """
    Extract text and metadata from HTML file using BeautifulSoup.
    Removes script, style, nav, footer. Preserves headings, paragraphs, lists, links, code.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    soup = BeautifulSoup(raw, "html.parser")

    # Metadata
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    author = ""
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        author = meta_author["content"]

    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"]

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    # Extract text with structure hints
    text_parts = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "td", "th"]):
        if element.name.startswith("h"):
            text_parts.append(f"\n[{element.name.upper()}] {element.get_text(' ', strip=True)}\n")
        elif element.name == "pre":
            text_parts.append(f"\n[CODE]\n{element.get_text(' ', strip=True)}\n[/CODE]\n")
        else:
            text_parts.append(element.get_text(' ', strip=True))

    text = "\n".join(text_parts)

    # Fallback: use get_text on body if empty
    if not text.strip():
        text = soup.get_text(separator="\n", strip=True)

    # Extract links
    links = []
    for a in soup.find_all("a", href=True):
        links.append({
            "href": a["href"],
            "text": a.get_text(strip=True),
        })

    metadata = {
        "title": title or filepath.stem,
        "author": author,
        "description": description,
        "source_url": "",  # could be derived from links or file path
        "links": links,
    }

    return {
        "text": text,
        "metadata": metadata,
        "format": "html",
    }