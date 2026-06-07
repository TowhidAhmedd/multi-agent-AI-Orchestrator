"""
Embedding model factory.

Free-tier strategy
------------------
* Primary   : HuggingFace sentence-transformers (fully local, no API key)
* Secondary : OpenAI text-embedding-3-small (when OPENAI_API_KEY is set)

The HuggingFace path uses `sentence-transformers/all-MiniLM-L6-v2` by default
which produces 384-dimensional embeddings and runs on CPU in <1 s per batch.
"""

import os
from functools import lru_cache
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings(use_openai: bool = False) -> Embeddings:
    """
    Return an Embeddings instance.

    Priority
    --------
    1. OpenAI  — when use_openai=True and OPENAI_API_KEY is set.
    2. HuggingFace sentence-transformers — free, local, always available.

    Args:
        use_openai: Force OpenAI embeddings.

    Returns:
        LangChain-compatible Embeddings object.
    """
    settings = get_settings()

    if use_openai or (settings.openai_api_key and settings.embedding_model.startswith("text-embedding")):
        if settings.openai_api_key:
            try:
                from langchain_openai import OpenAIEmbeddings

                logger.info("Using OpenAI embeddings: %s", settings.embedding_model)
                return OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key,
                )
            except ImportError:
                logger.warning("langchain-openai not installed; falling back to HuggingFace.")

    # HuggingFace local embeddings (free, no key needed)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        hf_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Using HuggingFace embeddings: %s", hf_model)
        return HuggingFaceEmbeddings(
            model_name=hf_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        pass

    # Ultimate fallback — fake embeddings for testing without any ML deps
    logger.warning("No embedding library found; using FakeEmbeddings (testing only).")
    from langchain_core.embeddings import FakeEmbeddings

    return FakeEmbeddings(size=384)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using the configured model.

    Args:
        texts: Strings to embed.

    Returns:
        List of embedding vectors.
    """
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Args:
        query: Query text.

    Returns:
        Embedding vector.
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(query)
