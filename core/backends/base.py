"""
Base class for backend providers.
"""
import requests


class BackendProvider:
    def __init__(self, config: dict):
        self.config = config
        self.url = config.get("url", "").rstrip("/")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "")

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(self, messages, model=None, max_tokens=1024, temperature=0.0, system=None):
        """Return assistant text reply."""
        raise NotImplementedError

    def embeddings(self, texts, model=None):
        """Return list of embedding vectors."""
        raise NotImplementedError

    def list_models(self):
        """Return list of model names (optional)."""
        raise NotImplementedError

    def health_check(self):
        """Return True if backend is reachable."""
        raise NotImplementedError
