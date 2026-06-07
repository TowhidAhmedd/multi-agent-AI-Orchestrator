"""
LLM factory — returns the appropriate chat model based on config.

Free-tier strategy
------------------
* Primary  : groq/llama-3.1-8b-instant  (very fast, generous free tier)
* Fallback : groq/mixtral-8x7b-32768
* OpenAI   : used only when OPENAI_API_KEY is set AND use_openai=True
"""

import os
from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel
from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)


def get_llm(
    temperature: float = 0.1,
    model_override: Optional[str] = None,
    use_openai: bool = False,
) -> BaseChatModel:
    """
    Return an instantiated chat model.

    Priority order
    --------------
    1. OpenAI  (when use_openai=True and OPENAI_API_KEY present)
    2. Groq    (when GROQ_API_KEY present  — default free-tier path)
    3. Ollama  (local fallback, no API key required)

    Args:
        temperature: Sampling temperature.
        model_override: Explicit model name; overrides settings.
        use_openai: Force OpenAI even when Groq key is available.

    Returns:
        Instantiated BaseChatModel.

    Raises:
        RuntimeError: When no usable provider is configured.
    """
    settings = get_settings()

    # ── OpenAI path ──────────────────────────────────────────────────────────
    if use_openai or (settings.openai_api_key and not settings.groq_api_key):
        if settings.openai_api_key:
            try:
                from langchain_openai import ChatOpenAI

                model = model_override or settings.llm_model
                logger.info("Using OpenAI model: %s", model)
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=settings.openai_api_key,
                    request_timeout=settings.request_timeout,
                    max_retries=settings.max_retries,
                )
            except ImportError:
                logger.warning("langchain-openai not installed; falling through.")

    # ── Groq path (default free-tier) ────────────────────────────────────────
    if settings.groq_api_key:
        try:
            from langchain_groq import ChatGroq

            model = model_override or settings.groq_model
            logger.info("Using Groq model: %s", model)
            return ChatGroq(
                model=model,
                temperature=temperature,
                api_key=settings.groq_api_key,
                max_retries=settings.max_retries,
            )
        except ImportError:
            logger.warning("langchain-groq not installed; falling through.")

    # ── Ollama local fallback ─────────────────────────────────────────────────
    try:
        from langchain_ollama import ChatOllama

        model = model_override or os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info("Using Ollama model: %s (local)", model)
        return ChatOllama(model=model, temperature=temperature)
    except ImportError:
        pass

    raise RuntimeError(
        "No LLM provider configured. "
        "Set GROQ_API_KEY (free), OPENAI_API_KEY, or run Ollama locally."
    )


@lru_cache(maxsize=4)
def get_cached_llm(temperature: float = 0.1) -> BaseChatModel:
    """Cached LLM instance to avoid redundant initialisation."""
    return get_llm(temperature=temperature)
