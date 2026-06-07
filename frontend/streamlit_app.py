"""
Streamlit frontend for the Multi-Agent AI Orchestrator.

Features
--------
* Chat interface with streaming-style responses
* File upload (PDF, TXT, Markdown) → instant ingestion
* Conversation history with role badges
* Source / citation display
* Agent activity viewer (collapsible)
* Workflow diagram
* Sidebar: upload, clear chat, system status
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 Multi-Agent AI Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API base URL ──────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_health() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_chat(query: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    payload = {"query": query, "messages": messages}
    r = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def api_upload(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    r = requests.post(
        f"{API_BASE}/upload",
        files={"file": (filename, file_bytes)},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def api_metrics() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def role_badge(role: str) -> str:
    return "👤" if role == "user" else "🤖"


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "last_metadata" not in st.session_state:
    st.session_state.last_metadata = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .chat-user   { background:#e8f0fe; border-radius:12px; padding:12px 16px; margin:4px 0; }
    .chat-assist { background:#f8f9fa; border-radius:12px; padding:12px 16px; margin:4px 0; }
    .source-card { background:#fff8e1; border-left:4px solid #f9a825;
                   border-radius:4px; padding:8px 12px; margin:4px 0; font-size:.85rem; }
    .metric-card { background:#e3f2fd; border-radius:8px; padding:10px; text-align:center; }
    .status-ok   { color:#2e7d32; font-weight:600; }
    .status-err  { color:#c62828; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 AI Orchestrator")
    st.caption("Multi-Agent LangGraph Pipeline")
    st.divider()

    # System status
    st.subheader("🟢 System Status")
    health = api_health()
    if health:
        st.markdown(f"**Status:** <span class='status-ok'>● Online</span>", unsafe_allow_html=True)
        providers = health.get("providers", {})
        st.write(f"🧠 LLM: `{providers.get('llm', 'unknown')}`")
        st.write(f"🗄️ Vector DB: `{providers.get('vector_store', 'unknown')}`")
        st.write(f"🔍 Search: `{providers.get('search', 'unknown')}`")
        st.write(f"📡 Tracing: `{providers.get('tracing', 'disabled')}`")
    else:
        st.markdown(
            "**Status:** <span class='status-err'>● API Offline</span>",
            unsafe_allow_html=True,
        )
        st.info(f"Start the FastAPI server at {API_BASE}")

    st.divider()

    # File upload
    st.subheader("📄 Upload Documents")
    uploaded = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        help="Files are ingested into the vector store for RAG retrieval.",
    )
    if uploaded is not None:
        if st.button("⬆️ Ingest Document", use_container_width=True):
            with st.spinner("Ingesting…"):
                try:
                    result = api_upload(uploaded.read(), uploaded.name)
                    if result.get("status") in ("success", "partial"):
                        st.success(
                            f"✅ {uploaded.name} ingested\n"
                            f"• {result.get('chunks_created', 0)} chunks\n"
                            f"• {result.get('vectors_upserted', 0)} vectors"
                        )
                        st.session_state.uploaded_files.append(uploaded.name)
                    else:
                        st.error(f"❌ {result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    if st.session_state.uploaded_files:
        st.caption("Ingested files:")
        for fname in st.session_state.uploaded_files:
            st.write(f"  📎 {fname}")

    st.divider()

    # Metrics
    st.subheader("📊 Live Metrics")
    metrics = api_metrics()
    if metrics:
        counters = metrics.get("counters", {})
        col1, col2 = st.columns(2)
        col1.metric("Requests", counters.get("api.chat_requests", 0))
        col2.metric("Uploads", counters.get("api.uploads", 0))

        lat = metrics.get("latency", {}).get("api.chat_latency", {})
        if lat:
            st.metric("Avg Latency", f"{lat.get('mean_ms', 0):.0f} ms")

        errors = metrics.get("errors", {})
        total_errors = sum(errors.values())
        if total_errors:
            st.warning(f"⚠️ {total_errors} error(s) recorded")
    else:
        st.caption("Metrics unavailable (API offline).")

    st.divider()

    # Controls
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.last_metadata = {}
        st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
col_chat, col_info = st.columns([3, 1])

with col_chat:
    st.title("🤖 Multi-Agent AI Orchestrator")
    st.caption(
        "Powered by **LangGraph** · **Pinecone RAG** · **Tavily Search** · **LangSmith Tracing**"
    )

    # Workflow diagram
    with st.expander("🔄 Workflow Architecture", expanded=False):
        st.markdown(
            """
```
START
  │
  ▼
[🧠 Planner Agent] ─► Analyse intent · Decide strategy
  │
  ▼ (conditional routing)
  ├──► [📚 Retrieval Agent] → Pinecone semantic search
  │         │
  │         ▼ (if search needed)
  └──► [🌐 Search Agent] → Tavily / DuckDuckGo web search
              │
              ▼
         [✍️ Synthesizer Agent] → Merge context · Generate answer
              │
             END
```
            """
        )

    # Chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

    # Chat input
    if prompt := st.chat_input("Ask me anything…"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("🤖 Multi-agent pipeline running…"):
                try:
                    t0 = time.time()
                    response = api_chat(prompt, st.session_state.messages[:-1])
                    elapsed = time.time() - t0

                    answer = response.get("answer", "No answer returned.")
                    sources = response.get("sources", [])
                    metadata = response.get("metadata", {})

                    st.markdown(answer)

                    # Agent activity summary
                    plan = metadata.get("plan", {})
                    if plan:
                        with st.expander("🔍 Agent Activity", expanded=False):
                            st.json(
                                {
                                    "planner_plan": plan,
                                    "rag_chunks_used": metadata.get("rag_chunks_used", 0),
                                    "search_results_used": metadata.get("search_results_used", 0),
                                    "synthesis_latency_s": metadata.get("synthesis_latency_s"),
                                    "total_elapsed_s": round(elapsed, 2),
                                }
                            )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.sources = sources
                    st.session_state.last_metadata = metadata

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Cannot reach the API server. "
                        f"Make sure FastAPI is running at {API_BASE}."
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")


with col_info:
    # Sources panel
    if st.session_state.sources:
        st.subheader("📎 Sources")
        for i, src in enumerate(st.session_state.sources, 1):
            src_type = src.get("type", "web")
            icon = "📚" if src_type == "rag" else "🌐"
            title = src.get("title", f"Source {i}")
            url = src.get("url", "")
            st.markdown(
                f"<div class='source-card'>{icon} <b>{title}</b>"
                + (f"<br><a href='{url}' target='_blank'>↗ link</a>" if url else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    # Last request metadata
    if st.session_state.last_metadata:
        st.subheader("ℹ️ Last Request")
        plan = st.session_state.last_metadata.get("plan", {})
        if plan:
            st.write(f"**Complexity:** `{plan.get('complexity', 'N/A')}`")
            st.write(f"**Used RAG:** {'✅' if plan.get('needs_rag') else '❌'}")
            st.write(f"**Used Search:** {'✅' if plan.get('needs_search') else '❌'}")
            st.write(
                f"**RAG Chunks:** {st.session_state.last_metadata.get('rag_chunks_used', 0)}"
            )
            st.write(
                f"**Search Results:** {st.session_state.last_metadata.get('search_results_used', 0)}"
            )

    # Quick-start tips
    with st.expander("💡 Quick-Start Tips", expanded=True):
        st.markdown(
            """
**Try asking:**
- "What is LangGraph and how does it work?"
- "Latest AI research trends in 2024"
- "Summarise the documents I uploaded"

**To use RAG:**
1. Upload a PDF/TXT via the sidebar
2. Ask questions about its content

**Free-tier providers:**
- 🧠 Groq (LLM) — fastest free tier
- 🗄️ Pinecone Starter (vectors)
- 🔍 Tavily (1k searches/month)
            """
        )
