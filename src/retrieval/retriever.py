"""
Retriever — wraps Pinecone vector search behind a clean interface.

When Pinecone is not configured, falls back to a simple in-memory keyword
search so the app remains runnable during local development.
"""

from typing import Any, Dict, List, Optional

from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.retrieval.embeddings import embed_query
from src.retrieval.pinecone_client import query_vectors

logger = get_logger(__name__)

# In-memory fallback store (populated by ingest when Pinecone is absent)
_local_store: List[Dict[str, Any]] = []


def add_to_local_store(records: List[Dict[str, Any]]) -> None:
    """Add vector records to the in-memory fallback store."""
    _local_store.extend(records)
    logger.debug("Local store now holds %d records.", len(_local_store))


def _keyword_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Simple token-overlap search against the local store."""
    query_tokens = set(query.lower().split())
    scored = []
    for rec in _local_store:
        text = rec.get("metadata", {}).get("text", "")
        tokens = set(text.lower().split())
        overlap = len(query_tokens & tokens) / max(len(query_tokens), 1)
        scored.append((overlap, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k] if _ > 0]


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    namespace: str = "default",
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Retrieve the most relevant text chunks for a query.

    Tries Pinecone first; falls back to in-memory keyword search.

    Args:
        query: Natural language query.
        top_k: Number of chunks to retrieve.
        namespace: Pinecone namespace.
        filter_dict: Optional metadata filter.

    Returns:
        List of text strings (the retrieved context chunks).
    """
    settings = get_settings()
    k = top_k or settings.top_k_results

    # ── Pinecone path ─────────────────────────────────────────────────────────
    if settings.pinecone_api_key:
        try:
            query_vec = embed_query(query)
            matches = query_vectors(
                query_vector=query_vec,
                top_k=k,
                namespace=namespace,
                filter_dict=filter_dict,
            )
            if matches:
                texts = [m["metadata"].get("text", "") for m in matches if m.get("metadata")]
                logger.info("Pinecone returned %d chunks for query.", len(texts))
                return texts
        except Exception as exc:
            logger.warning("Pinecone retrieval error: %s — falling back.", exc)

    # ── In-memory fallback ────────────────────────────────────────────────────
    if _local_store:
        results = _keyword_search(query, top_k=k)
        texts = [r.get("metadata", {}).get("text", "") for r in results]
        logger.info("Local store returned %d chunks (fallback).", len(texts))
        return texts

    logger.warning("No vector store available — retrieval returning empty.")
    return []


def retrieve_with_scores(
    query: str,
    top_k: Optional[int] = None,
    namespace: str = "default",
) -> List[Dict[str, Any]]:
    """
    Like retrieve() but returns metadata and scores as well.

    Returns:
        List of {text, source, score} dicts.
    """
    settings = get_settings()
    k = top_k or settings.top_k_results

    if settings.pinecone_api_key:
        try:
            query_vec = embed_query(query)
            matches = query_vectors(query_vector=query_vec, top_k=k, namespace=namespace)
            return [
                {
                    "text": m["metadata"].get("text", ""),
                    "source": m["metadata"].get("source", "unknown"),
                    "score": round(m.get("score", 0.0), 4),
                }
                for m in matches
            ]
        except Exception as exc:
            logger.warning("Pinecone scored retrieval failed: %s", exc)

    return []
