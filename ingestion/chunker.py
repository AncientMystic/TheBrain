from core.text_utils import chunk_text
import config


def chunk_document(text: str) -> list[str]:
    """
    Split entire document text into overlapping chunks.
    Uses chunk_size and overlap from config.
    """
    return chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)