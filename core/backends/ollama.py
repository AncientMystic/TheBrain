"""
Ollama provider (native API and OpenAI-compatible).
"""
import requests
from core.backends.base import BackendProvider
import logging
logger = logging.getLogger(__name__)


class Provider(BackendProvider):
    def __init__(self, config):
        super().__init__(config)
        # If backend is ollama_openai, url already points to /v1
        self.mode = config.get("backend", "ollama")
        if self.mode == "ollama":
            self.base_url = self.url.rstrip("/") + "/api"
        else:
            self.base_url = self.url.rstrip("/") + "/v1"

    def chat(self, messages, model=None, max_tokens=1024, temperature=0.0, system=None):
        model = model or self.model
        if self.mode == "ollama":
            # native Ollama API
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt = msg["content"] + "\n\n" + prompt
                elif msg["role"] == "user":
                    prompt += msg["content"] + "\n"
            url = f"{self.base_url}/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=480)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        else:
            # OpenAI-compatible
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=480)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def embeddings(self, texts, model=None):
        model = model or self.config.get("embeddings_model", self.model)
        if self.mode == "ollama":
            # Try native batch embeddings endpoint first
            try:
                url = f"{self.base_url}/embed"
                payload = {"model": model, "input": texts}
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=240)
                resp.raise_for_status()
                data = resp.json()
                embeddings = data.get("embeddings")
                if embeddings is not None and len(embeddings) == len(texts):
                    return embeddings
            except Exception:
                logger.warning("Unexpected exception occurred", exc_info=True)
                pass

            # Fallback to sequential /api/embeddings
            embeddings = []
            for text in texts:
                url = f"{self.base_url}/embeddings"
                payload = {"model": model, "prompt": text}
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=240)
                resp.raise_for_status()
                data = resp.json()
                embeddings.append(data["embedding"])
            return embeddings
        else:
            url = f"{self.base_url}/embeddings"
            payload = {"input": texts, "model": model}
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=240)
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

    def list_models(self):
        if self.mode == "ollama":
            resp = requests.get(f"{self.base_url}/tags", headers=self._headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        else:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]

    def loaded_models(self):
        """Names of models currently resident (no reload needed). Native Ollama
        only (/api/ps); other modes return None (unknown, not False)."""
        try:
            if self.mode != "ollama":
                return None
            import requests
            resp = requests.get(f"{self.base_url}/ps", headers=self._headers(), timeout=5)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return None

    def health_check(self):
        try:
            if self.mode == "ollama":
                resp = requests.get(f"{self.base_url}/tags", headers=self._headers(), timeout=5)
            else:
                resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=5)
            return resp.status_code == 200
        except Exception:
            logger.warning("Unexpected exception occurred", exc_info=True)
            return False
