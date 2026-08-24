"""
Backend provider abstraction for TheBrain.
Supports LM Studio, Ollama, Kobold.cpp, and generic OpenAI-compatible backends.
"""
import importlib

BACKEND_TYPES = {
    "lmstudio": "lmstudio",
    "ollama": "ollama",
    "ollama_openai": "ollama_openai",
    "koboldcpp": "koboldcpp",
    "openai_compatible": "openai_compatible",
}


def create_backend(endpoint_config):
    """
    Return a backend provider instance from an endpoint configuration dict.
    Raises ValueError for unsupported backend types.
    """
    backend_type = endpoint_config.get("backend", "lmstudio").lower()
    if backend_type not in BACKEND_TYPES:
        raise ValueError(
            f"Unsupported backend type: {backend_type}. "
            f"Supported types: {', '.join(BACKEND_TYPES.keys())}"
        )

    module_name = f"core.backends.{BACKEND_TYPES[backend_type]}"
    module = importlib.import_module(module_name)
    provider_class = getattr(module, "Provider")
    return provider_class(endpoint_config)
