"""
Document ingestion pipeline.

Supports: PDF, TXT, Markdown.
Chunks documents, generates embeddings, and upserts into Pinecone
(or a local FAISS store when Pinecone is not configured).
"""

import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.retrieval.embeddings import embed_texts
from src.retrieval.pinecone_client import upsert_vectors

logger = get_logger(__name__)


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load_pdf(path: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        return loader.load()
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return []


def _load_txt(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": path})]


def _load_markdown(path: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        loader = UnstructuredMarkdownLoader(path)
        return loader.load()
    except ImportError:
        return _load_txt(path)  # plain-text fallback


LOADERS = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".md": _load_markdown,
    ".markdown": _load_markdown,
}


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from disk using the appropriate loader.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError: When the file extension is unsupported.
    """
    ext = Path(file_path).suffix.lower()
    loader_fn = LOADERS.get(ext)
    if loader_fn is None:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS)}")

    logger.info("Loading document: %s", file_path)
    docs = loader_fn(file_path)
    logger.info("Loaded %d page(s) from %s", len(docs), file_path)
    return docs


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split documents into overlapping text chunks.

    Args:
        documents: Source documents to chunk.
        chunk_size: Characters per chunk (defaults to settings).
        chunk_overlap: Overlap between chunks (defaults to settings).

    Returns:
        List of chunked Document objects.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks from %d documents.", len(chunks), len(documents))
    return chunks


# ─── Vector records ───────────────────────────────────────────────────────────

def _doc_id(text: str, source: str) -> str:
    """Generate a deterministic ID based on content hash."""
    digest = hashlib.md5(f"{source}:{text[:200]}".encode()).hexdigest()[:12]
    return f"doc-{digest}"


def build_vector_records(chunks: List[Document]) -> List[Dict[str, Any]]:
    """
    Convert Document chunks into Pinecone-ready vector records.

    Args:
        chunks: Chunked documents.

    Returns:
        List of {id, values, metadata} dicts.
    """
    texts = [c.page_content for c in chunks]
    logger.info("Generating embeddings for %d chunks…", len(texts))
    embeddings = embed_texts(texts)

    records = []
    for chunk, embedding in zip(chunks, embeddings):
        doc_id = _doc_id(chunk.page_content, chunk.metadata.get("source", ""))
        records.append(
            {
                "id": doc_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content[:1000],  # Pinecone metadata cap
                    "source": chunk.metadata.get("source", "unknown"),
                    "page": chunk.metadata.get("page", 0),
                },
            }
        )
    return records


# ─── Main ingestion entry-point ───────────────────────────────────────────────

def ingest_file(file_path: str, namespace: str = "default") -> Dict[str, Any]:
    """
    Full ingestion pipeline for a single file.

    Args:
        file_path: Path to the document.
        namespace: Pinecone namespace for the vectors.

    Returns:
        Summary dict with ingestion stats.
    """
    start = time.perf_counter()

    try:
        docs = load_document(file_path)
        if not docs:
            return {"status": "error", "message": "No content extracted.", "file": file_path}

        chunks = chunk_documents(docs)
        records = build_vector_records(chunks)
        success = upsert_vectors(records, namespace=namespace)

        elapsed = round(time.perf_counter() - start, 2)
        return {
            "status": "success" if success else "partial",
            "file": file_path,
            "pages_loaded": len(docs),
            "chunks_created": len(chunks),
            "vectors_upserted": len(records),
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", file_path, exc)
        return {"status": "error", "message": str(exc), "file": file_path}


def ingest_directory(directory: str, namespace: str = "default") -> List[Dict[str, Any]]:
    """
    Ingest all supported files in a directory.

    Args:
        directory: Path to the directory.
        namespace: Pinecone namespace.

    Returns:
        List of per-file ingestion results.
    """
    results = []
    for fname in os.listdir(directory):
        ext = Path(fname).suffix.lower()
        if ext in LOADERS:
            result = ingest_file(os.path.join(directory, fname), namespace=namespace)
            results.append(result)
    return results
