"""Endpoint routing for different model groups."""
import itertools
import threading
import config

_small_pool = []
_large_pool = []
_chat_pool = []
_pools_built = False
_lock = threading.Lock()
_cycles = {}

def _build_pools():
    global _pools_built
    if _pools_built:
        return
    with _lock:
        if _pools_built:
            return

        small = []
        # Primary small model
        if getattr(config, "SMALL_MODEL_URL", "") and getattr(config, "SMALL_MODEL_NAME", ""):
            small.append({"url": config.SMALL_MODEL_URL, "model": config.SMALL_MODEL_NAME, "api_key": "not-needed"})
        # Secondary small model
        if getattr(config, "SMALL_MODEL_URL_2", "") and getattr(config, "SMALL_MODEL_NAME_2", ""):
            small.append({"url": config.SMALL_MODEL_URL_2, "model": config.SMALL_MODEL_NAME_2, "api_key": "not-needed"})

        large = []
        if getattr(config, "LARGE_MODEL_URL", "") and getattr(config, "LARGE_MODEL_NAME", ""):
            large.append({"url": config.LARGE_MODEL_URL, "model": config.LARGE_MODEL_NAME, "api_key": "not-needed"})

        chat = []
        if getattr(config, "USE_CHAT_MODEL", False) and getattr(config, "CHAT_MODEL_URL", "") and getattr(config, "CHAT_MODEL_NAME", ""):
            chat.append({"url": config.CHAT_MODEL_URL, "model": config.CHAT_MODEL_NAME, "api_key": "not-needed"})

        # Fallback to main endpoints
        main = list(config.LLM_ENDPOINTS)

        if not small:
            small = main
        if not large:
            large = main
        if not chat:
            chat = main

        _small_pool.extend(small)
        _large_pool.extend(large)
        _chat_pool.extend(chat)

        _cycles["small"] = itertools.cycle(_small_pool)
        _cycles["large"] = itertools.cycle(_large_pool)
        _cycles["chat"] = itertools.cycle(_chat_pool)
        _pools_built = True

def get_endpoint_for_group(group: str) -> dict:
    """Return an endpoint for the given group ('small', 'large', 'chat', 'main')."""
    _build_pools()
    if group == "small":
        return next(_cycles["small"])
    elif group == "large":
        return next(_cycles["large"])
    elif group == "chat":
        return next(_cycles["chat"])
    else:
        return next(itertools.cycle(config.LLM_ENDPOINTS)) if config.LLM_ENDPOINTS else None

def get_chat_endpoint():
    return get_endpoint_for_group("chat")

def get_extraction_endpoint():
    group = getattr(config, "EXTRACTION_MODEL_GROUP", "small")
    return get_endpoint_for_group(group)

def get_validation_endpoint():
    group = getattr(config, "VALIDATION_MODEL_GROUP", "large")
    return get_endpoint_for_group(group)
