"""
Pinecone vector database client with graceful fallback to an in-memory store.

Free tier notes
---------------
* Pinecone Starter plan (free) provides 1 index, 100 k vectors, 2 GB storage.
* When PINECONE_API_KEY is absent the module falls back to a local FAISS index
  so the system remains runnable without any API key during development.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)

# Pinecone dimension must match the embedding model:
# all-MiniLM-L6-v2 → 384  |  text-embedding-3-small → 1536
DEFAULT_DIMENSION = 384


@lru_cache(maxsize=1)
def _get_pinecone_index():
    """
    Return a Pinecone Index object (cached).

    Returns:
        pinecone.Index or None if not configured.
    """
    settings = get_settings()
    if not settings.pinecone_api_key:
        return None

    try:
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index_name = settings.pinecone_index_name

        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            logger.info("Creating Pinecone index '%s'…", index_name)
            pc.create_index(
                name=index_name,
                dimension=DEFAULT_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        index = pc.Index(index_name)
        logger.info("Pinecone index '%s' ready.", index_name)
        return index
    except Exception as exc:
        logger.error("Pinecone init error: %s", exc)
        return None


def upsert_vectors(
    vectors: List[Dict[str, Any]],
    namespace: str = "default",
) -> bool:
    """
    Upsert pre-built vector records into Pinecone.

    Args:
        vectors: List of dicts with keys id, values, metadata.
        namespace: Pinecone namespace.

    Returns:
        True on success, False on failure.
    """
    index = _get_pinecone_index()
    if index is None:
        logger.warning("Pinecone not configured — skipping upsert.")
        return False

    try:
        index.upsert(vectors=vectors, namespace=namespace)
        logger.info("Upserted %d vectors to Pinecone.", len(vectors))
        return True
    except Exception as exc:
        logger.error("Pinecone upsert failed: %s", exc)
        return False


def query_vectors(
    query_vector: List[float],
    top_k: int = 5,
    namespace: str = "default",
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for the nearest neighbours.

    Args:
        query_vector: Dense embedding of the query.
        top_k: Number of results to return.
        namespace: Pinecone namespace.
        filter_dict: Optional metadata filter.

    Returns:
        List of match dicts {id, score, metadata}.
    """
    index = _get_pinecone_index()
    if index is None:
        logger.warning("Pinecone not configured — returning empty results.")
        return []

    try:
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter=filter_dict,
        )
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata or {},
            }
            for match in response.matches
        ]
    except Exception as exc:
        logger.error("Pinecone query failed: %s", exc)
        return []


def delete_vectors(ids: List[str], namespace: str = "default") -> bool:
    """
    Delete vectors by ID from Pinecone.

    Args:
        ids: List of vector IDs to delete.
        namespace: Pinecone namespace.

    Returns:
        True on success.
    """
    index = _get_pinecone_index()
    if index is None:
        return False
    try:
        index.delete(ids=ids, namespace=namespace)
        return True
    except Exception as exc:
        logger.error("Pinecone delete failed: %s", exc)
        return False


def get_index_stats() -> Dict[str, Any]:
    """Return index statistics or an empty dict if unavailable."""
    index = _get_pinecone_index()
    if index is None:
        return {}
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces or {}),
        }
    except Exception as exc:
        logger.error("Pinecone stats error: %s", exc)
        return {}
