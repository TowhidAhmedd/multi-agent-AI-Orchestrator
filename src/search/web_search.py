"""
Web search module.

Primary  : Tavily Search API (free tier: 1 000 searches/month)
Fallback : DuckDuckGo (completely free, no API key)
"""

import time
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


# ─── Tavily ───────────────────────────────────────────────────────────────────

def _tavily_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search using the Tavily API.

    Args:
        query: Search string.
        max_results: Maximum number of results.

    Returns:
        List of {title, url, snippet} dicts.
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY not set.")

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
        )
        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:500],
                }
            )
        logger.info("Tavily returned %d results for: %s", len(results), query)
        return results
    except ImportError:
        raise RuntimeError("tavily-python not installed. Run: pip install tavily-python")


# ─── DuckDuckGo fallback ──────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search using DuckDuckGo (no API key required).

    Args:
        query: Search string.
        max_results: Maximum number of results.

    Returns:
        List of {title, url, snippet} dicts.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")[:500],
                    }
                )
        logger.info("DuckDuckGo returned %d results for: %s", len(results), query)
        return results
    except ImportError:
        raise RuntimeError("duckduckgo-search not installed. Run: pip install duckduckgo-search")
    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)
        return []


# ─── Public API ───────────────────────────────────────────────────────────────

def web_search(
    query: str,
    max_results: int = 5,
    force_ddg: bool = False,
) -> List[Dict[str, Any]]:
    """
    Execute a web search using Tavily with DuckDuckGo as fallback.

    Args:
        query: The search query string.
        max_results: Maximum results to return.
        force_ddg: Skip Tavily and go straight to DuckDuckGo.

    Returns:
        List of {title, url, snippet} result dicts.
    """
    settings = get_settings()

    if not force_ddg and settings.tavily_api_key:
        try:
            return _tavily_search(query, max_results=max_results)
        except Exception as exc:
            logger.warning("Tavily failed (%s) — falling back to DuckDuckGo.", exc)

    return _ddg_search(query, max_results=max_results)


def multi_search(
    queries: List[str],
    max_results_per_query: int = 3,
) -> List[Dict[str, Any]]:
    """
    Run multiple search queries and merge/deduplicate results.

    Args:
        queries: List of search strings.
        max_results_per_query: Results per query.

    Returns:
        Deduplicated list of results sorted by query order.
    """
    seen_urls: set = set()
    all_results: List[Dict[str, Any]] = []

    for query in queries:
        try:
            results = web_search(query, max_results=max_results_per_query)
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    r["query"] = query
                    all_results.append(r)
            time.sleep(0.5)  # polite delay between requests
        except Exception as exc:
            logger.warning("Search failed for '%s': %s", query, exc)

    return all_results


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Convert search results to a readable string for LLM context.

    Args:
        results: List of result dicts.

    Returns:
        Formatted string.
    """
    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', 'No title')}")
        lines.append(f"    URL: {r.get('url', '')}")
        lines.append(f"    {r.get('snippet', '')}")
        lines.append("")
    return "\n".join(lines)
