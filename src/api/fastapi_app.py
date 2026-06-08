"""
FastAPI application — REST interface for the multi-agent orchestrator.
Render.com compatible — reads PORT from environment variable.
"""

import asyncio
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.logging_config import get_logger, setup_logging
from src.config.settings import get_settings
from src.graph.graph_builder import run_workflow
from src.monitoring.metrics import Timer, get_snapshot, increment, record_error
from src.observability.langsmith_tracing import init_langsmith
from src.retrieval.ingest import ingest_file

logger = get_logger(__name__)
settings = get_settings()

setup_logging(level=settings.log_level)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Non-blocking startup — avoids Render health-check timeout (502)."""
    port = os.environ.get("PORT", "8000")
    logger.info("Starting %s v%s on port %s", settings.app_name, settings.app_version, port)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, init_langsmith)

    yield

    executor.shutdown(wait=False, cancel_futures=True)
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready Multi-Agent AI Orchestrator powered by LangGraph.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    messages: Optional[List[Dict[str, str]]] = Field(default=[])
    session_id: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    session_id: str


class UploadResponse(BaseModel):
    status: str
    file: str
    pages_loaded: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Render health check endpoint — must return 200 quickly."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": time.time(),
        "port": os.environ.get("PORT", "8000"),
        "providers": {
            "llm": "groq" if settings.groq_api_key else (
                "openai" if settings.openai_api_key else "ollama"
            ),
            "vector_store": "pinecone" if settings.pinecone_api_key else "local",
            "search": "tavily" if settings.tavily_api_key else "duckduckgo",
            "tracing": "langsmith" if settings.langsmith_api_key else "disabled",
        },
    }


@app.get("/", tags=["System"])
async def root() -> Dict[str, str]:
    """Root endpoint — confirms API is running."""
    return {
        "message": "Multi-Agent AI Orchestrator is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/metrics", tags=["System"])
async def get_metrics() -> Dict[str, Any]:
    return get_snapshot()


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the multi-agent workflow and return the final answer."""
    session_id = request.session_id or str(uuid.uuid4())
    increment("api.chat_requests")
    logger.info("Chat request [%s]: %.80s…", session_id, request.query)

    with Timer("api.chat_latency"):
        try:
            loop = asyncio.get_event_loop()
            state = await loop.run_in_executor(
                executor,
                lambda: run_workflow(
                    query=request.query,
                    messages=request.messages or [],
                ),
            )
        except Exception as exc:
            record_error("api.chat")
            logger.error("Workflow error [%s]: %s", session_id, exc)
            raise HTTPException(status_code=500, detail=f"Workflow error: {exc}")

    answer = state.get("final_answer", "No answer generated.")
    sources = state.get("sources", [])
    metadata = {
        **state.get("metadata", {}),
        "session_id": session_id,
        "plan": state.get("execution_plan", {}),
    }

    return ChatResponse(
        answer=answer,
        sources=sources,
        metadata=metadata,
        session_id=session_id,
    )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload and ingest a document into the vector store."""
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}
    ext = Path(file.filename or "file.txt").suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    dest_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save error: {exc}")

    increment("api.uploads")
    result = ingest_file(str(dest_path))

    if result.get("status") == "error":
        return UploadResponse(
            status="error",
            file=file.filename or "",
            message=result.get("message", "Unknown error"),
        )

    return UploadResponse(
        status=result.get("status", "success"),
        file=file.filename or str(dest_path),
        pages_loaded=result.get("pages_loaded", 0),
        chunks_created=result.get("chunks_created", 0),
        vectors_upserted=result.get("vectors_upserted", 0),
        elapsed_seconds=result.get("elapsed_seconds", 0.0),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.api.fastapi_app:app",
        host="0.0.0.0",
        port=port,
        log_level=settings.log_level.lower(),
    )
