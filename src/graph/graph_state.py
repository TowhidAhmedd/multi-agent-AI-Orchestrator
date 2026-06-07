"""
LangGraph state schema — shared across all graph nodes.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """
    Typed state object threaded through every LangGraph node.

    Fields
    ------
    query : str
        Original user query.
    execution_plan : dict
        JSON plan produced by the Planner agent.
    rag_context : list[str]
        Document chunks retrieved from Pinecone.
    search_results : list[dict]
        Snippets returned by the Search agent.
    final_answer : str
        Synthesised response sent back to the user.
    sources : list[dict]
        Citations attached to the final answer.
    error : str | None
        Any error message to surface upstream.
    metadata : dict
        Arbitrary per-request metadata (latency, token counts …).
    messages : list[dict]
        Conversation history (role / content pairs).
    file_paths : list[str]
        Paths of uploaded files queued for ingestion.
    """

    query: str
    execution_plan: Dict[str, Any]
    rag_context: List[str]
    search_results: List[Dict[str, Any]]
    final_answer: str
    sources: List[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]
    messages: List[Dict[str, str]]
    file_paths: List[str]


def create_initial_state(query: str, messages: Optional[List[Dict[str, str]]] = None) -> AgentState:
    """
    Build a fresh state dict for a new request.

    Args:
        query: The user's question.
        messages: Optional prior conversation turns.

    Returns:
        Populated AgentState ready to enter the graph.
    """
    return AgentState(
        query=query,
        execution_plan={},
        rag_context=[],
        search_results=[],
        final_answer="",
        sources=[],
        error=None,
        metadata={},
        messages=messages or [],
        file_paths=[],
    )
