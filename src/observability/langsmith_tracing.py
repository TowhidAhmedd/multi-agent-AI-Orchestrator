"""
LangSmith observability utilities — tracing, metadata tagging, run feedback.
"""

import functools
import time
from typing import Any, Callable, Dict, Optional

from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


def init_langsmith() -> bool:
    """
    Verify LangSmith is reachable and tracing is enabled.

    Returns:
        True if LangSmith tracing is active, False otherwise.
    """
    settings = get_settings()
    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_API_KEY not set — tracing disabled.")
        return False

    try:
        settings.configure_langsmith()
        logger.info("LangSmith tracing enabled → project: %s", settings.langsmith_project)
        return True
    except Exception as exc:
        logger.warning("LangSmith init failed: %s", exc)
        return False


def trace_agent(agent_name: str) -> Callable:
    """
    Decorator that wraps an agent function with a LangSmith run span.

    Usage::

        @trace_agent("planner")
        def run_planner(state):
            ...

    Args:
        agent_name: Human-readable label shown in the LangSmith UI.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug("Agent '%s' completed in %.3fs", agent_name, elapsed)
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error(
                    "Agent '%s' failed after %.3fs: %s", agent_name, elapsed, exc
                )
                raise

        return wrapper

    return decorator


class WorkflowTracer:
    """
    Lightweight wrapper around LangSmith for manual span management.

    Example::

        tracer = WorkflowTracer("my-run-id")
        tracer.start_span("planner")
        ...
        tracer.end_span("planner", outputs={"plan": ...})
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._spans: Dict[str, float] = {}
        self._client = self._get_client()

    def _get_client(self):
        settings = get_settings()
        if not settings.langsmith_api_key:
            return None
        try:
            from langsmith import Client

            return Client(api_key=settings.langsmith_api_key)
        except Exception:
            return None

    def start_span(self, name: str) -> None:
        """Record the start time of a named span."""
        self._spans[name] = time.perf_counter()
        logger.debug("[Trace %s] → %s started", self.run_id, name)

    def end_span(self, name: str, outputs: Optional[Dict[str, Any]] = None) -> float:
        """
        Close a span and return its duration in seconds.

        Args:
            name: Span name matching a prior start_span call.
            outputs: Optional key/value outputs to log.

        Returns:
            Duration in seconds, or 0.0 if span not found.
        """
        start = self._spans.pop(name, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        logger.debug(
            "[Trace %s] ← %s finished (%.3fs) %s",
            self.run_id,
            name,
            elapsed,
            outputs or "",
        )
        return elapsed

    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log arbitrary metadata for this workflow run."""
        logger.info("[Trace %s] metadata: %s", self.run_id, metadata)


def get_tracer(run_id: str) -> WorkflowTracer:
    """Factory for WorkflowTracer instances."""
    return WorkflowTracer(run_id=run_id)
