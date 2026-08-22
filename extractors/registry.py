import importlib
from pathlib import Path


EXTRACTOR_REGISTRY = {
    ".pdf": ("pdf_extractor", "extract_pdf"),
    ".html": ("html_extractor", "extract_html"),
    ".htm": ("html_extractor", "extract_html"),
    ".md": ("markdown_extractor", "extract_markdown"),
    ".markdown": ("markdown_extractor", "extract_markdown"),
    ".txt": ("text_extractor", "extract_text"),
    ".text": ("text_extractor", "extract_text"),
    ".docx": ("docx_extractor", "extract_docx"),
    ".epub": ("epub_extractor", "extract_epub"),
    ".rtf": ("rtf_extractor", "extract_rtf"),
    ".ipynb": ("ipynb_extractor", "extract_ipynb"),
    ".py": ("source_code_extractor", "extract_source_code"),
    ".js": ("source_code_extractor", "extract_source_code"),
    ".ts": ("source_code_extractor", "extract_source_code"),
    ".java": ("source_code_extractor", "extract_source_code"),
    ".c": ("source_code_extractor", "extract_source_code"),
    ".cpp": ("source_code_extractor", "extract_source_code"),
    ".h": ("source_code_extractor", "extract_source_code"),
    ".hpp": ("source_code_extractor", "extract_source_code"),
    ".cs": ("source_code_extractor", "extract_source_code"),
    ".go": ("source_code_extractor", "extract_source_code"),
    ".rb": ("source_code_extractor", "extract_source_code"),
    ".php": ("source_code_extractor", "extract_source_code"),
    ".swift": ("source_code_extractor", "extract_source_code"),
    ".sh": ("source_code_extractor", "extract_source_code"),
    ".bat": ("source_code_extractor", "extract_source_code"),
    ".ps1": ("source_code_extractor", "extract_source_code"),
    ".json": ("source_code_extractor", "extract_source_code"),
    ".xml": ("source_code_extractor", "extract_source_code"),
    ".csv": ("source_code_extractor", "extract_source_code"),
    ".yaml": ("source_code_extractor", "extract_source_code"),
    ".yml": ("source_code_extractor", "extract_source_code"),
    ".toml": ("source_code_extractor", "extract_source_code"),
    ".ini": ("source_code_extractor", "extract_source_code"),
    ".log": ("source_code_extractor", "extract_source_code"),
    ".rst": ("source_code_extractor", "extract_source_code"),
    ".tex": ("source_code_extractor", "extract_source_code"),
    ".adoc": ("source_code_extractor", "extract_source_code"),
}


def get_extractor(extension):
    if not extension.startswith("."):
        extension = "." + extension
    entry = EXTRACTOR_REGISTRY.get(extension.lower())
    if not entry:
        return None
    module_name, func_name = entry
    module = importlib.import_module(f"extractors.{module_name}")
    return getattr(module, func_name)


def extract_text_from_file(filepath):
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    extractor = get_extractor(ext)
    if not extractor:
        raise ValueError(f"Unsupported file extension: {ext}")
    result = extractor(filepath)
    if "text" not in result:
        result["text"] = ""
    if "metadata" not in result:
        result["metadata"] = {}
    result["format"] = ext.lstrip(".")
    return result