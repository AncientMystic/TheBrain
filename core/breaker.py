"""Endpoint circuit breaker: fail fast on dead backends, auto-recover.

In-memory only (no schema, no persistence): consecutive call failures against
an endpoint key open the circuit for a cooldown; one success closes it again.
Timeout values elsewhere are untouched — this only skips the wait for an
endpoint that has already proven itself dead. Default on, config-tunable.
"""
import time
import threading

_lock = threading.Lock()
_state = {}  # (url, model) -> {"failures": int, "opened_until": float}


def _cfg(name, default):
    try:
        import config as _c
        return getattr(_c, name, default)
    except Exception:
        return default


def _key(endpoint):
    try:
        return (str(endpoint.get("url", "")), str(endpoint.get("model", "")))
    except Exception:
        return ("", "")


def is_open(endpoint):
    """True when the circuit is open (skip this endpoint right now)."""
    if not _cfg("BREAKER_ENABLED", True):
        return False
    k = _key(endpoint)
    with _lock:
        s = _state.get(k)
        if not s:
            return False
        if time.monotonic() < s.get("opened_until", 0.0):
            return True
        return False  # cooldown elapsed: half-open, allow one probe call


def record_success(endpoint):
    k = _key(endpoint)
    with _lock:
        _state.pop(k, None)


def record_failure(endpoint):
    """Count one failed call; open the circuit at threshold. Returns open-now."""
    if not _cfg("BREAKER_ENABLED", True):
        return False
    try:
        threshold = max(1, int(_cfg("BREAKER_FAILURE_THRESHOLD", 3)))
        cooldown = max(1.0, float(_cfg("BREAKER_COOLDOWN_SECONDS", 300)))
    except Exception:
        threshold, cooldown = 3, 300.0
    k = _key(endpoint)
    with _lock:
        s = _state.setdefault(k, {"failures": 0, "opened_until": 0.0})
        s["failures"] = int(s.get("failures", 0)) + 1
        if s["failures"] >= threshold:
            s["opened_until"] = time.monotonic() + cooldown
            try:
                from core.metrics import inc_counter as _inc
                _inc("breaker_opened_total")
            except Exception:
                pass
            return True
        return False


def reset(endpoint=None):
    """Clear state (tests + operator recovery). None resets all."""
    with _lock:
        if endpoint is None:
            _state.clear()
        else:
            _state.pop(_key(endpoint), None)
