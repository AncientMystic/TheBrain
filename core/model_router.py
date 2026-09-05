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
_cycle_lengths = {}
_main_len = -1

def _reset_pools():
    global _pools_built, _main_len
    del _small_pool[:]
    del _large_pool[:]
    del _chat_pool[:]
    _cycles.clear()
    _cycle_lengths.clear()
    _pools_built = False
    _main_len = -1

def _build_pools():
    global _pools_built, _main_len
    if _pools_built:
        return
    with _lock:
        if _pools_built:
            return

        small = []
        if getattr(config, "SMALL_MODEL_URL", "") and getattr(config, "SMALL_MODEL_NAME", ""):
            small.append({"url": config.SMALL_MODEL_URL, "model": config.SMALL_MODEL_NAME, "api_key": "not-needed"})
        if getattr(config, "SMALL_MODEL_URL_2", "") and getattr(config, "SMALL_MODEL_NAME_2", ""):
            small.append({"url": config.SMALL_MODEL_URL_2, "model": config.SMALL_MODEL_NAME_2, "api_key": "not-needed"})

        large = []
        if getattr(config, "LARGE_MODEL_URL", "") and getattr(config, "LARGE_MODEL_NAME", ""):
            large.append({"url": config.LARGE_MODEL_URL, "model": config.LARGE_MODEL_NAME, "api_key": "not-needed"})

        chat = []
        if getattr(config, "USE_CHAT_MODEL", False) and getattr(config, "CHAT_MODEL_URL", "") and getattr(config, "CHAT_MODEL_NAME", ""):
            chat.append({"url": config.CHAT_MODEL_URL, "model": config.CHAT_MODEL_NAME, "api_key": "not-needed"})

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
        _cycle_lengths["small"] = len(_small_pool)
        _cycle_lengths["large"] = len(_large_pool)
        _cycle_lengths["chat"] = len(_chat_pool)
        _main_len = len(config.LLM_ENDPOINTS)
        _pools_built = True

def get_endpoint_for_group(group: str) -> dict:
    """Return an endpoint for the given group ('small', 'large', 'chat', 'main')."""
    _build_pools()
    # Rebuild pools if endpoint configuration changed (compare list lengths,
    # never len() of a cycle object). next() on a cycle is thread-safe enough
    # for rotation purposes; pool rebuilds take the lock briefly.
    _lock.acquire()
    try:
        pools = {"small": _small_pool, "large": _large_pool, "chat": _chat_pool}
        stale = _cycle_lengths.get(group, -1) != len(pools.get(group, [])) if group in pools else False
        if _main_len != len(config.LLM_ENDPOINTS):
            stale = True
        if stale:
            _reset_pools()
            rebuild = True
        else:
            rebuild = False
        cycle = None if rebuild else _cycles.get(group)
    finally:
        _lock.release()
    if rebuild:
        _build_pools()
        with _lock:
            cycle = _cycles.get(group)
    if group in ("small", "large", "chat"):
        return next(cycle)
    with _lock:
        eps = list(config.LLM_ENDPOINTS)
    return next(itertools.cycle(eps)) if eps else None

def get_chat_endpoint():
    return get_endpoint_for_group("chat")

def get_extraction_endpoint():
    group = getattr(config, "EXTRACTION_MODEL_GROUP", "small")
    return get_endpoint_for_group(group)

def get_validation_endpoint():
    group = getattr(config, "VALIDATION_MODEL_GROUP", "large")
    return get_endpoint_for_group(group)
