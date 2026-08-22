from pathlib import Path


def extract_source_code(filepath: Path) -> dict:
    """
    Extract text from source code or generic text-like file.
    For code, we preserve line structure; no comment stripping by default.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Determine language from extension
    ext = filepath.suffix.lower()
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".sh": "shell",
        ".bat": "batch",
        ".ps1": "powershell",
    }
    language = language_map.get(ext, "text")

    # Optional: if it's code, we might want to extract comments too, but here just text.

    return {
        "text": text,
        "metadata": {
            "title": filepath.stem,
            "author": "",
            "language": language,
        },
        "format": "source",
    }