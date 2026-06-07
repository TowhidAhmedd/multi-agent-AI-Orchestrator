"""
LangGraph node wrappers.

Each function here adapts an agent function to the LangGraph node contract:
  - Receives the full AgentState dict
  - Returns a (partial) state dict with updated fields
"""

from typing import Any, Dict

from src.agents.planner_agent import run_planner
from src.agents.retrieval_agent import run_retrieval
from src.agents.search_agent import run_search
from src.agents.synthesizer_agent import run_synthesizer
from src.config.logging_config import get_logger
from src.graph.graph_state import AgentState

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Node functions
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Planner Agent.

    Analyses user intent and produces an execution plan.
    """
    logger.debug("→ Entering planner node.")
    result = run_planner(state)
    logger.debug("← Planner node complete.")
    return result


def retrieval_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Retrieval Agent.

    Fetches relevant chunks from the Pinecone vector store.
    """
    logger.debug("→ Entering retrieval node.")
    result = run_retrieval(state)
    logger.debug("← Retrieval node complete.")
    return result


def search_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Search Agent.

    Executes web searches and processes results.
    """
    logger.debug("→ Entering search node.")
    result = run_search(state)
    logger.debug("← Search node complete.")
    return result


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: Synthesizer Agent.

    Merges all context and generates the final answer.
    """
    logger.debug("→ Entering synthesizer node.")
    result = run_synthesizer(state)
    logger.debug("← Synthesizer node complete.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def route_after_planner(state: AgentState) -> str:
    """
    Conditional edge: decide which node runs after the Planner.

    Routing logic:
    - If both RAG and search are needed → "retrieval" (retrieval runs first,
      then search, then synthesizer — serial for simplicity)
    - If only RAG → "retrieval"
    - If only search → "search"
    - Otherwise → "synthesizer" (answer from LLM knowledge alone)

    Returns:
        Node name string.
    """
    plan = state.get("execution_plan", {})
    needs_rag = plan.get("needs_rag", True)
    needs_search = plan.get("needs_search", True)

    if needs_rag:
        logger.debug("Router: needs_rag=True → retrieval")
        return "retrieval"
    elif needs_search:
        logger.debug("Router: needs_rag=False, needs_search=True → search")
        return "search"
    else:
        logger.debug("Router: neither rag nor search → synthesizer")
        return "synthesizer"


def route_after_retrieval(state: AgentState) -> str:
    """
    Conditional edge: decide which node runs after Retrieval.

    If search is also needed, route to search; otherwise go to synthesizer.

    Returns:
        Node name string.
    """
    plan = state.get("execution_plan", {})
    if plan.get("needs_search", True):
        logger.debug("Router (post-retrieval): needs_search=True → search")
        return "search"
    logger.debug("Router (post-retrieval): needs_search=False → synthesizer")
    return "synthesizer"
