"""
Retrieval Agent — fetches relevant context from the Pinecone vector store.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser

from src.config.logging_config import get_logger
from src.llm.llm_factory import get_llm
from src.llm.prompts import retrieval_prompt
from src.monitoring.metrics import Timer, increment, record_error
from src.observability.langsmith_tracing import trace_agent
from src.retrieval.retriever import retrieve

logger = get_logger(__name__)


def _parse_retrieval_output(raw: str, fallback_chunks: List[str]) -> List[str]:
    """
    Parse the LLM's ranked/filtered chunk list.

    Falls back to raw retrieved chunks if parsing fails.

    Args:
        raw: Raw LLM output string.
        fallback_chunks: Original unfiltered chunks.

    Returns:
        List of text strings.
    """
    text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            chunks = data.get("relevant_chunks", [])
            if chunks:
                return chunks
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse retrieval output — using raw chunks.")
    return fallback_chunks


@trace_agent("retrieval")
def run_retrieval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the Retrieval agent.

    Uses the execution plan's `rag_query` to fetch chunks from Pinecone,
    then optionally re-ranks them with an LLM call.

    Args:
        state: Current graph state.

    Returns:
        Updated state with `rag_context` populated.
    """
    plan = state.get("execution_plan", {})

    # If the planner said RAG is not needed, skip
    if not plan.get("needs_rag", True):
        logger.info("Planner flagged needs_rag=False — skipping retrieval.")
        return {**state, "rag_context": []}

    rag_query = plan.get("rag_query") or state.get("query", "")
    increment("retrieval.calls")

    with Timer("retrieval.latency") as t:
        try:
            # 1. Vector retrieval
            raw_chunks = retrieve(rag_query, top_k=8)

            if not raw_chunks:
                logger.info("No RAG chunks retrieved for query: %s", rag_query)
                return {**state, "rag_context": []}

            # 2. LLM re-ranking / filtering (optional — skipped if > 5 chunks already)
            if len(raw_chunks) > 5:
                try:
                    llm = get_llm(temperature=0.0)
                    chain = retrieval_prompt | llm | StrOutputParser()
                    chunks_text = "\n---\n".join(raw_chunks[:10])
                    raw_output = chain.invoke(
                        {"query": rag_query, "chunks": chunks_text}
                    )
                    ranked_chunks = _parse_retrieval_output(raw_output, raw_chunks)
                except Exception as exc:
                    logger.warning("LLM re-ranking failed: %s — using raw order.", exc)
                    ranked_chunks = raw_chunks
            else:
                ranked_chunks = raw_chunks

            logger.info(
                "Retrieval agent returned %d chunks in %.2fs.",
                len(ranked_chunks),
                t.elapsed,
            )
            return {**state, "rag_context": ranked_chunks[:5]}

        except Exception as exc:
            record_error("retrieval")
            logger.error("Retrieval agent error: %s", exc)
            return {**state, "rag_context": [], "error": str(exc)}
