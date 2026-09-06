"""Probe LM Studio server-side queueing: latency vs concurrency sweep.

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/probe_lm_queue.py
Sends tiny completions (max_tokens=8) at concurrency 1/2/4 against the
primary endpoint and reports wall/p50/max. Flat wall with rising concurrency
means the server parallelizes (client workers justified); linear wall growth
means server-side serialization (raising client workers only adds timeouts).

Bounded: 7 requests total. Safe to re-run whenever the backend changes.
"""
import os
import sys
import time
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
import config  # noqa: E402

EP = config.LLM_ENDPOINTS[0]
BASE = str(EP.get("url", "")).rstrip("/")
MODEL = str(EP.get("model", ""))


def one(_):
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{BASE}/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 8, "temperature": 0.0}, timeout=300)
        ok = r.status_code == 200
    except Exception as e:
        print(f"  (request failed: {e})")
        ok = False
    return (time.perf_counter() - t0, ok)


def main():
    try:
        r = requests.get(f"{BASE}/models", timeout=10)
        print(f"liveness: {r.status_code} ({BASE} model={MODEL})")
    except Exception as e:
        print(f"SERVER-OFFLINE: {e}")
        return 0
    for level in (1, 2, 4):
        with cf.ThreadPoolExecutor(max_workers=level) as ex:
            t0 = time.perf_counter()
            res = list(ex.map(one, range(level)))
            wall = time.perf_counter() - t0
        lats = sorted(dt for dt, _ in res)
        oks = sum(1 for _, ok in res if ok)
        print(f"concurrency={level} wall={wall * 1000:.0f}ms "
              f"min={lats[0] * 1000:.0f}ms p50={lats[len(lats) // 2] * 1000:.0f}ms "
              f"max={lats[-1] * 1000:.0f}ms ok={oks}/{level}")
    print("PROBE-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
