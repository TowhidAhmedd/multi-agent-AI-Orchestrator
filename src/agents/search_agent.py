"""
Search Agent — fetches real-time web information using Tavily / DuckDuckGo.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser

from src.config.logging_config import get_logger
from src.llm.llm_factory import get_llm
from src.llm.prompts import search_prompt
from src.monitoring.metrics import Timer, increment, record_error
from src.observability.langsmith_tracing import trace_agent
from src.search.web_search import format_search_results, multi_search

logger = get_logger(__name__)


def _parse_search_output(raw: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM's processed search results JSON.

    Args:
        raw: Raw LLM output string.

    Returns:
        List of processed result dicts, or empty list on failure.
    """
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("processed_results", [])
        except json.JSONDecodeError:
            pass
    return []


@trace_agent("search")
def run_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the Search agent.

    Reads `execution_plan.search_queries`, runs web searches,
    and uses an LLM to filter/process the results.

    Args:
        state: Current graph state.

    Returns:
        Updated state with `search_results` and `sources` populated.
    """
    plan = state.get("execution_plan", {})

    # If the planner said search is not needed, skip
    if not plan.get("needs_search", True):
        logger.info("Planner flagged needs_search=False — skipping web search.")
        return {**state, "search_results": []}

    queries = plan.get("search_queries") or [state.get("query", "")]
    increment("search.calls")

    with Timer("search.latency") as t:
        try:
            # 1. Execute web searches
            raw_results = multi_search(queries, max_results_per_query=4)

            if not raw_results:
                logger.info("No web search results found.")
                return {**state, "search_results": []}

            # 2. LLM processing / filtering
            formatted = format_search_results(raw_results[:10])
            try:
                llm = get_llm(temperature=0.0)
                chain = search_prompt | llm | StrOutputParser()
                raw_output = chain.invoke(
                    {
                        "query": state.get("query", ""),
                        "search_results": formatted,
                    }
                )
                processed = _parse_search_output(raw_output)
            except Exception as exc:
                logger.warning("LLM search processing failed: %s — using raw results.", exc)
                processed = [
                    {
                        "snippet": r.get("snippet", ""),
                        "source": r.get("url", ""),
                        "relevance": 0.8,
                    }
                    for r in raw_results[:5]
                ]

            # Build sources list for citation
            sources = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "type": "web",
                }
                for r in raw_results[:5]
                if r.get("url")
            ]

            logger.info(
                "Search agent returned %d processed results in %.2fs.",
                len(processed),
                t.elapsed,
            )
            return {**state, "search_results": processed, "sources": sources}

        except Exception as exc:
            record_error("search")
            logger.error("Search agent error: %s", exc)
            return {**state, "search_results": [], "error": str(exc)}
