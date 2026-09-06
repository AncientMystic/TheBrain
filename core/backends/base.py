"""
Base class for backend providers.
"""
import requests
import logging
logger = logging.getLogger(__name__)


def stream_openai_sse(url, payload, headers, timeout=480):
    """Yield content deltas from an OpenAI-compatible SSE stream."""
    with requests.post(url, json=payload, headers=headers,
                       timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                import json as _json
                obj = _json.loads(data)
                delta = obj["choices"][0].get("delta", {}).get("content", "")
            except Exception:
                continue
            if delta:
                yield delta


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

    def chat_stream(self, messages, model=None, max_tokens=1024, temperature=0.0, system=None):
        """Yield assistant text deltas (true streaming where supported).

        Default: single yield of the blocking chat() result, so every
        backend streams correctly on day one; OpenAI-compatible backends
        override with real token streaming.
        """
        yield self.chat(messages, model=model, max_tokens=max_tokens,
                        temperature=temperature, system=system)

    def embeddings(self, texts, model=None):
        """Return list of embedding vectors."""
        raise NotImplementedError

    def list_models(self):
        """Return list of model names (optional)."""
        raise NotImplementedError

    def health_check(self):
        """Return True if backend is reachable."""
        raise NotImplementedError
