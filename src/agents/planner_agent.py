"""
Planner Agent — analyses user intent and produces an execution plan.
"""

import json
import re
import time
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser

from src.config.logging_config import get_logger
from src.llm.llm_factory import get_llm
from src.llm.prompts import planner_prompt
from src.monitoring.metrics import Timer, increment, record_error
from src.observability.langsmith_tracing import trace_agent

logger = get_logger(__name__)

_DEFAULT_PLAN: Dict[str, Any] = {
    "needs_search": True,
    "needs_rag": True,
    "complexity": "medium",
    "search_queries": [],
    "rag_query": "",
    "reasoning": "Default plan — LLM parsing failed.",
}


def _parse_plan(raw: str) -> Dict[str, Any]:
    """
    Extract JSON from the raw LLM output.

    Handles markdown code fences and stray text around the JSON block.

    Args:
        raw: Raw LLM response string.

    Returns:
        Parsed plan dict, or a sensible default on parse failure.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find the outermost JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse planner output; using default plan.")
    return _DEFAULT_PLAN.copy()


@trace_agent("planner")
def run_planner(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the Planner agent.

    Reads `state["query"]` and `state["messages"]`, produces an
    execution plan and writes it back to `state["execution_plan"]`.

    Args:
        state: Current graph state.

    Returns:
        Updated state dict.
    """
    query = state.get("query", "")
    messages = state.get("messages", [])

    # Format last 3 conversation turns for context
    history_lines = []
    for msg in messages[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_lines.append(f"{role.upper()}: {content[:200]}")
    history = "\n".join(history_lines) if history_lines else "No prior conversation."

    increment("planner.calls")

    with Timer("planner.latency") as t:
        try:
            llm = get_llm(temperature=0.0)
            chain = planner_prompt | llm | StrOutputParser()
            raw_output = chain.invoke({"query": query, "history": history})
            plan = _parse_plan(raw_output)

            # Ensure rag_query falls back to the original query if empty
            if not plan.get("rag_query"):
                plan["rag_query"] = query

            # Ensure at least one search query if search is needed
            if plan.get("needs_search") and not plan.get("search_queries"):
                plan["search_queries"] = [query]

            logger.info(
                "Planner produced plan (complexity=%s, search=%s, rag=%s) in %.2fs",
                plan.get("complexity"),
                plan.get("needs_search"),
                plan.get("needs_rag"),
                t.elapsed,
            )

            return {**state, "execution_plan": plan}

        except Exception as exc:
            record_error("planner")
            logger.error("Planner agent error: %s", exc)
            fallback_plan = {
                **_DEFAULT_PLAN,
                "rag_query": query,
                "search_queries": [query],
            }
            return {**state, "execution_plan": fallback_plan, "error": str(exc)}
