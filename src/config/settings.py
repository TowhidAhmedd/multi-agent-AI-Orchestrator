"""
Application settings loaded from environment variables.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration via environment variables."""

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"

    # Tavily Search
    tavily_api_key: str = ""

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "ai-orchestrator"
    pinecone_environment: str = "gcp-starter"

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "ai-orchestrator"
    langchain_tracing_v2: str = "true"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # # Groq / HuggingFace
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # # Frontend / API base
    api_base_url: str = "http://localhost:8000"

    # App
    app_name: str = "Multi-Agent AI Orchestrator"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5

    # Retry
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def configure_langsmith(self) -> None:
        """Set LangSmith env vars for auto-tracing."""
        if self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = self.langchain_tracing_v2
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project

    def configure_openai(self) -> None:
        """Set OpenAI env var."""
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings singleton."""
    settings = Settings()
    settings.configure_langsmith()
    # settings.configure_openai()
    return settings


