"""
Lightweight in-process metrics — no Prometheus, no Grafana.
Thread-safe counters and histograms backed by a plain dict.
"""

import statistics
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()

_counters: Dict[str, int] = defaultdict(int)
_latencies: Dict[str, List[float]] = defaultdict(list)
_errors: Dict[str, int] = defaultdict(int)
_start_time: float = time.time()


# ─── Public API ───────────────────────────────────────────────────────────────

def increment(name: str, value: int = 1) -> None:
    """Increment a named counter."""
    with _lock:
        _counters[name] += value


def record_latency(name: str, seconds: float) -> None:
    """Append a latency sample (seconds) to a named histogram."""
    with _lock:
        _latencies[name].append(seconds)
        # Keep rolling window of 1 000 samples
        if len(_latencies[name]) > 1000:
            _latencies[name] = _latencies[name][-1000:]


def record_error(name: str) -> None:
    """Increment the error counter for a named component."""
    with _lock:
        _errors[name] += 1
    logger.debug("Error recorded for: %s", name)


def get_snapshot() -> Dict[str, Any]:
    """
    Return a full metrics snapshot suitable for the /metrics endpoint.

    Returns:
        Dict with counters, latency stats, error counts, and uptime.
    """
    with _lock:
        latency_stats: Dict[str, Dict[str, float]] = {}
        for key, samples in _latencies.items():
            if samples:
                latency_stats[key] = {
                    "count": len(samples),
                    "mean_ms": round(statistics.mean(samples) * 1000, 2),
                    "median_ms": round(statistics.median(samples) * 1000, 2),
                    "min_ms": round(min(samples) * 1000, 2),
                    "max_ms": round(max(samples) * 1000, 2),
                    "p95_ms": round(
                        sorted(samples)[int(len(samples) * 0.95)] * 1000, 2
                    ),
                }

        return {
            "uptime_seconds": round(time.time() - _start_time, 1),
            "counters": dict(_counters),
            "latency": latency_stats,
            "errors": dict(_errors),
        }


def reset() -> None:
    """Reset all metrics (useful in tests)."""
    with _lock:
        _counters.clear()
        _latencies.clear()
        _errors.clear()


class Timer:
    """
    Context-manager / decorator for timing code blocks.

    Usage::

        with Timer("rag_retrieval") as t:
            docs = retriever.get_relevant_documents(query)
        print(t.elapsed)
    """

    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
            record_latency(self.metric_name, self.elapsed)
