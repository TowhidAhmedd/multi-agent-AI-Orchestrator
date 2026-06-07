"""
Synthesizer Agent — merges RAG context and web search results into a
final, citation-rich answer.
"""

from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser

from src.config.logging_config import get_logger
from src.llm.llm_factory import get_llm
from src.llm.prompts import synthesizer_prompt
from src.monitoring.metrics import Timer, increment, record_error
from src.observability.langsmith_tracing import trace_agent

logger = get_logger(__name__)


def _format_rag_context(chunks: List[str]) -> str:
    """Convert a list of RAG chunks into a readable string block."""
    if not chunks:
        return "No knowledge base context available."
    return "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )


def _format_search_results(results: List[Dict[str, Any]]) -> str:
    """Convert processed search results into a readable string block."""
    if not results:
        return "No web search results available."
    lines = []
    for i, r in enumerate(results, 1):
        source = r.get("source", "")
        snippet = r.get("snippet", "")
        lines.append(f"[Web {i}] {snippet}")
        if source:
            lines.append(f"  Source: {source}")
        lines.append("")
    return "\n".join(lines)


def _format_history(messages: List[Dict[str, str]], max_turns: int = 4) -> str:
    """Format recent conversation history."""
    if not messages:
        return "No prior conversation."
    recent = messages[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")[:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


@trace_agent("synthesizer")
def run_synthesizer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the Synthesizer agent.

    Reads `rag_context`, `search_results`, and `messages` from state,
    calls the LLM to produce a final answer, and writes it to
    `state["final_answer"]`.

    Args:
        state: Current graph state.

    Returns:
        Updated state with `final_answer` and merged `sources`.
    """
    query = state.get("query", "")
    rag_context = state.get("rag_context", [])
    search_results = state.get("search_results", [])
    messages = state.get("messages", [])
    existing_sources = state.get("sources", [])

    increment("synthesizer.calls")

    with Timer("synthesizer.latency") as t:
        try:
            rag_text = _format_rag_context(rag_context)
            search_text = _format_search_results(search_results)
            history_text = _format_history(messages)

            llm = get_llm(temperature=0.3)  # slight creativity for synthesis
            chain = synthesizer_prompt | llm | StrOutputParser()

            answer = chain.invoke(
                {
                    "query": query,
                    "rag_context": rag_text,
                    "search_results": search_text,
                    "history": history_text,
                }
            )

            # Add RAG sources (file-based) to the sources list
            rag_sources = [
                {"title": f"Knowledge Base Chunk {i + 1}", "type": "rag", "url": ""}
                for i in range(len(rag_context))
            ]
            all_sources = existing_sources + rag_sources

            logger.info(
                "Synthesizer produced %d-char answer in %.2fs.",
                len(answer),
                t.elapsed,
            )

            return {
                **state,
                "final_answer": answer,
                "sources": all_sources,
                "metadata": {
                    **state.get("metadata", {}),
                    "rag_chunks_used": len(rag_context),
                    "search_results_used": len(search_results),
                    "answer_length": len(answer),
                    "synthesis_latency_s": round(t.elapsed, 3),
                },
            }

        except Exception as exc:
            record_error("synthesizer")
            logger.error("Synthesizer agent error: %s", exc)

            # Emergency fallback answer
            fallback = (
                f"I encountered an error while generating a response: {exc}\n\n"
                "Please try again or rephrase your question."
            )
            return {**state, "final_answer": fallback, "error": str(exc)}
