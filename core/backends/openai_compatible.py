"""
Generic OpenAI-compatible provider with API-key auth.
Suitable for OpenAI, Groq, OpenRouter, Mistral, etc.
"""
import requests
from core.backends.base import BackendProvider
import logging
logger = logging.getLogger(__name__)


class Provider(BackendProvider):
    def chat(self, messages, model=None, max_tokens=1024, temperature=0.0, system=None):
        url = f"{self.url}/chat/completions"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=480)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def chat_stream(self, messages, model=None, max_tokens=1024, temperature=0.0, system=None):
        """True token streaming over OpenAI-compatible SSE."""
        from core.backends.base import stream_openai_sse
        url = f"{self.url}/chat/completions"
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        try:
            yield from stream_openai_sse(url, payload, self._headers())
        except Exception as e:
            logger.warning(f"Stream failed, falling back to blocking chat: {e}")
            yield self.chat(messages, model=model, max_tokens=max_tokens,
                            temperature=temperature, system=system)

    def embeddings(self, texts, model=None):
        url = f"{self.url}/embeddings"
        payload = {
            "input": texts,
            "model": model or self.config.get("embeddings_model", self.model),
        }
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=240)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def list_models(self):
        resp = requests.get(f"{self.url}/models", headers=self._headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def health_check(self):
        try:
            resp = requests.get(f"{self.url}/models", headers=self._headers(), timeout=5)
            return resp.status_code == 200
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            return False
