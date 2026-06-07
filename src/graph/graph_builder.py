"""
LangGraph workflow builder.

Constructs the StateGraph with 4 agent nodes and conditional routing.

Graph topology
--------------

    START
      │
      ▼
  [planner] ──(route_after_planner)──► [retrieval] ──(route_after_retrieval)──► [synthesizer]
                                                                                      │
                                       [search] ◄────────────────────────────────────┘
                                           │
                                           ▼
                                      [synthesizer]
                                           │
                                          END
"""

from functools import lru_cache
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.graph.graph_state import AgentState
from src.graph.nodes import (
    planner_node,
    retrieval_node,
    route_after_planner,
    route_after_retrieval,
    search_node,
    synthesizer_node,
)

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Returns:
        A compiled LangGraph application ready to invoke.
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("search", search_node)
    graph.add_node("synthesizer", synthesizer_node)

    # ── Entry edge ────────────────────────────────────────────────────────────
    graph.add_edge(START, "planner")

    # ── Conditional routing after planner ────────────────────────────────────
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retrieval": "retrieval",
            "search": "search",
            "synthesizer": "synthesizer",
        },
    )

    # ── Conditional routing after retrieval ───────────────────────────────────
    graph.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "search": "search",
            "synthesizer": "synthesizer",
        },
    )

    # ── Search always feeds into synthesizer ──────────────────────────────────
    graph.add_edge("search", "synthesizer")

    # ── Synthesizer is the terminal node ──────────────────────────────────────
    graph.add_edge("synthesizer", END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled


@lru_cache(maxsize=1)
def get_graph():
    """Return a cached compiled graph (singleton per process)."""
    return build_graph()


def run_workflow(
    query: str,
    messages: Optional[list] = None,
    file_paths: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Execute the full multi-agent workflow for a single query.

    Args:
        query: The user's question.
        messages: Optional conversation history.
        file_paths: Optional list of newly uploaded file paths.

    Returns:
        Final state dict containing `final_answer`, `sources`, `metadata`.
    """
    from src.graph.graph_state import create_initial_state
    from src.monitoring.metrics import Timer, increment

    increment("workflow.total_requests")

    initial_state = create_initial_state(
        query=query,
        messages=messages or [],
    )
    if file_paths:
        initial_state["file_paths"] = file_paths

    with Timer("workflow.end_to_end"):
        graph = get_graph()
        final_state = graph.invoke(initial_state)

    logger.info(
        "Workflow complete | answer_len=%d | sources=%d",
        len(final_state.get("final_answer", "")),
        len(final_state.get("sources", [])),
    )
    return final_state
