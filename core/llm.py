import random
import json
import re
import time
import threading
import itertools

import requests
import aiohttp
import asyncio

from core.backends import create_backend

import threading
_requests_local = threading.local()

def _get_requests_session():
    if not hasattr(_requests_local, "session"):
        s = requests.Session()
        try:
            import config as _cfg
            from requests.adapters import HTTPAdapter as _HA
            from urllib3.util.retry import Retry as _Retry
            _pool_n = int(getattr(_cfg, "LLM_POOL_CONNECTIONS", 10))
            _pool_m = int(getattr(_cfg, "LLM_POOL_MAXSIZE", 10))
            _retries = _Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
            _ad = _HA(pool_connections=_pool_n, pool_maxsize=_pool_m, max_retries=_retries)
            s.mount("http://", _ad)
            s.mount("https://", _ad)
            s.headers.update({"Connection": "keep-alive"})
        except Exception:
            pass
        _requests_local.session = s
    return _requests_local.session

def _select_endpoint_by_type(endpoint_type=None):
    """Return endpoint based on type ('small', 'large', 'audit', 'chat') if configured.

    Chat answers previously fell through to the shared global cycle, so consecutive
    messages could land on different models/servers and force weight reloads.
    Routing chat to its own pool keeps one conversation on one loaded model.
    """
    if endpoint_type == "small" and config.SMALL_MODEL_ENDPOINT:
        return config.SMALL_MODEL_ENDPOINT
    if endpoint_type == "large" and config.LARGE_MODEL_ENDPOINT:
        return config.LARGE_MODEL_ENDPOINT
    if endpoint_type == "audit" and config.AUDIT_MODEL_ENDPOINT:
        return config.AUDIT_MODEL_ENDPOINT
    if endpoint_type == "chat":
        if getattr(config, "CHAT_MODEL_ENDPOINT", None):
            return config.CHAT_MODEL_ENDPOINT
        try:
            from core.model_router import get_chat_endpoint
            return get_chat_endpoint()
        except Exception:
            pass
    return None



import config
from core.model_router import get_endpoint_for_group
from core.logger import get_logger
logger = get_logger(__name__)

_llm_cycle = None
_llm_cycle_len = -1
_llm_lock = threading.Lock()

def _get_next_llm_endpoint():
    global _llm_cycle, _llm_cycle_len
    with _llm_lock:
        current_len = len(config.LLM_ENDPOINTS)
        if _llm_cycle is None or current_len != _llm_cycle_len:
            _llm_cycle = itertools.cycle(config.LLM_ENDPOINTS)
            _llm_cycle_len = current_len
            if config.DEBUG_VERBOSE:
                print(f"    [LLM cycle] endpoints order: {[ep.get('url') + ':' + ep.get('model', '') for ep in config.LLM_ENDPOINTS]}")
        return next(_llm_cycle)

def call_model(prompt, model=None, max_tokens=1024, temperature=None,
               system="You are a helpful assistant.", endpoint=None, endpoint_type=None):
    if endpoint is None:
        selected = _select_endpoint_by_type(endpoint_type)
        if selected:
            endpoint = selected
            model = endpoint["model"]
        elif model:
            for ep in config.LLM_ENDPOINTS:
                if ep["model"] == model:
                    endpoint = ep
                    break
            if not endpoint:
                endpoint = config.LLM_ENDPOINTS[0]
        else:
            endpoint = _get_next_llm_endpoint()
            model = endpoint["model"]
    else:
        model = endpoint["model"]

    if config.DEBUG_VERBOSE:
        logger.debug(f"LLM call -> {endpoint['url']} model={model}")
        print(f"    [LLM endpoint] {endpoint['url']} model={model}")

    if temperature is None:
        temperature = 0.0

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if config.USE_JSON_MODE:
        payload["response_format"] = {"type": "json_object"}

    # Build backend provider once for this endpoint
    backend_provider = create_backend(endpoint)
    for attempt in range(config.API_RETRY_ATTEMPTS):
        try:
            output = backend_provider.chat(
                payload["messages"],
                model=model,
                max_tokens=payload.get("max_tokens", 1024),
                temperature=payload.get("temperature", 0.0),
                system=system,
            )
            if output:
                cleaned = re.sub(r'<thinking>.*?</thinking>', '', output, flags=re.DOTALL)
                cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL)
                return cleaned.strip()
            else:
                print(f"    (Empty model response, attempt {attempt+1})")
                if config.DEBUG_VERBOSE:
                    logger.warning(f"Empty model response, attempt {attempt+1}")
        except Exception as e:
            if config.DEBUG_VERBOSE:
                logger.exception(f"LLM exception: {e}")
        time.sleep(config.API_RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5))
    return ""


def repair_json(raw: str) -> str:
    """Repair common JSON malformations from LLM outputs."""
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?', '', raw)
    raw = re.sub(r'```$', '', raw).strip()
    raw = raw.rstrip()
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    # Insert missing commas between adjacent JSON values (using capture groups)
    raw = re.sub(r'\}\s*\{', '},{', raw)
    raw = re.sub(r'\]\s*\[', '],[', raw)
    raw = re.sub(r'\}\s*"', '},"', raw)
    raw = re.sub(r'\]\s*"', '],"', raw)
    raw = re.sub(r'\}\s*(\d)', r'},\1', raw)  # capture digit
    raw = re.sub(r'\]\s*(\d)', r'],\1', raw)  # capture digit

    # Ensure any trailing incomplete string is closed
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        raw += '"'

    # Balance braces/brackets
    open_braces = raw.count('{')
    close_braces = raw.count('}')
    open_brackets = raw.count('[')
    close_brackets = raw.count(']')
    while open_braces > close_braces:
        raw += '}'
        close_braces += 1
    while open_brackets > close_brackets:
        raw += ']'
        close_brackets += 1
    return raw

def extract_first_json(raw: str):
    """Attempt to extract the first complete JSON object/array from raw text."""
    raw = raw.strip()
    if not raw:
        return None

    # Remove markdown fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw).strip()

    decoder = json.JSONDecoder()
    for i in range(len(raw)):
        ch = raw[i]
        if ch in ('{', '['):
            try:
                obj, _ = decoder.raw_decode(raw[i:])
                return obj
            except json.JSONDecodeError as e:
                logger.debug(f"Handled exception: {e}", exc_info=True)
                continue
    return None


def call_model_json(prompt, model=None, max_tokens=4096, temperature=None,
                    system="You are a meticulous assistant that returns only valid JSON.",
                    unwrap_list=True, endpoint=None, endpoint_type=None):
    def _parse(raw):
        if not raw:
            return None

        parsed = extract_first_json(raw)
        if parsed is not None:
            if unwrap_list and isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    return parsed[0]
                return None
            return parsed

        repaired = repair_json(raw)
        try:
            parsed = json.loads(repaired)
            if unwrap_list and isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    return parsed[0]
                return None
            return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"Handled exception: {e}", exc_info=True)
            pass

        return None

    raw = call_model(prompt, model=model, max_tokens=max_tokens,
                     temperature=temperature, system=system, endpoint=endpoint, endpoint_type=endpoint_type)
    parsed = _parse(raw)
    if parsed is not None:
        return parsed

    # Retry once with larger max_tokens to avoid truncation.
    larger_max = int(max_tokens * 1.5) + 512
    raw2 = call_model(prompt, model=model, max_tokens=larger_max,
                      temperature=temperature, system=system, endpoint=endpoint, endpoint_type=endpoint_type)
    parsed2 = _parse(raw2)
    if parsed2 is not None:
        return parsed2

    preview = (raw or "")[:500].replace("\n", " ")
    if config.DEBUG_VERBOSE:
        print(f"    (JSON parse failure preview: {preview})")
    if config.DEBUG_VERBOSE:
        logger.error("Failed to parse JSON from LLM response")
        logger.debug(f"Raw response (first 500 chars): {raw[:500]}")
    return None
