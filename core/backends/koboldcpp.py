"""
Kobold.cpp provider (OpenAI-compatible chat).
Embeddings may fallback to configured embeddings_url or raise.
"""
import requests
from core.backends.base import BackendProvider
import logging
logger = logging.getLogger(__name__)


class Provider(BackendProvider):
    def __init__(self, config):
        super().__init__(config)
        self.chat_url = config.get("chat_url", self.url.rstrip("/") + "/v1")
        self.embeddings_url = config.get("embeddings_url", self.url.rstrip("/") + "/v1")

    def chat(self, messages, model=None, max_tokens=1024, temperature=0.0, _system=None):
        url = f"{self.chat_url}/chat/completions"
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

    def embeddings(self, texts, model=None):
        url = f"{self.embeddings_url}/embeddings"
        payload = {
            "input": texts,
            "model": model or self.config.get("embeddings_model", self.model),
        }
        resp = requests.post(url, json=payload, headers=self._headers(), timeout=240)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def list_models(self):
        resp = requests.get(f"{self.chat_url}/models", headers=self._headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def health_check(self):
        try:
            resp = requests.get(f"{self.chat_url}/models", headers=self._headers(), timeout=5)
            return resp.status_code == 200
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            return False
