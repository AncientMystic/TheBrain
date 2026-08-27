"""
Simple in-memory metrics registry with Prometheus exposition.
"""
from collections import defaultdict
import time
import threading

_lock = threading.Lock()

_counters = defaultdict(int)
_histograms = defaultdict(list)  # store values


def inc_counter(name, amount=1):
    with _lock:
        _counters[name] += amount


def observe_histogram(name, value):
    with _lock:
        _histograms[name].append(value)


def get_counter(name):
    with _lock:
        return _counters.get(name, 0)


def get_histogram(name):
    with _lock:
        return list(_histograms.get(name, []))


def get_all_metrics():
    """Return all metrics in Prometheus text format."""
    lines = []
    with _lock:
        for name, value in _counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        for name, values in _histograms.items():
            if not values:
                continue
            lines.append(f"# TYPE {name} histogram")
            count = len(values)
            avg = sum(values) / count if count else 0
            lines.append(f"{name}_count {count}")
            lines.append(f"{name}_avg {avg}")
    return "\n".join(lines)


class Timer:
    """Context manager to record duration in a histogram."""
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start
        observe_histogram(self.metric_name, duration)
