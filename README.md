# 🤖 Multi-Agent AI Orchestrator

> Production-ready multi-agent AI system built with **LangGraph**, **LangChain**, **Pinecone**, **LangSmith**, **FastAPI**, and **Streamlit**.  
> Fully **free-tier friendly** — runs with Groq LLM, HuggingFace embeddings, Pinecone Starter, Tavily, and LangSmith free plans.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Multi-Agent AI Orchestrator                      │
│                                                                      │
│   ┌──────────────┐    ┌────────────────────────────────────────────┐ │
│   │  Streamlit   │    │              LangGraph Workflow            │ │
│   │   Frontend   │───►│                                            │ │
│   └──────────────┘    │  START                                     │ │
│                       │    │                                       │ │
│   ┌──────────────┐    │    ▼                                       │ │
│   │   FastAPI    │    │  [🧠 Planner Agent]                        │ │
│   │   Backend    │    │    │  Analyse intent · Decide strategy     │ │
│   └──────────────┘    │    │                                       │ │
│                       │    ▼ (conditional routing)                 │ │
│   ┌──────────────┐    │    ├──► [📚 Retrieval Agent]               │ │
│   │  LangSmith   │    │    │         │  Pinecone semantic search   │ │
│   │  Tracing     │    │    │         │                             │ │
│   └──────────────┘    │    │         ▼ (if search also needed)     │ │
│                       │    └──► [🌐 Search Agent]                  │ │
│   ┌──────────────┐    │              │  Tavily / DuckDuckGo        │ │
│   │  Pinecone    │    │              │                             │ │
│   │  Vector DB   │    │              ▼                             │ │
│   └──────────────┘    │         [✍️ Synthesizer Agent]             │ │
│                       │              │  Merge · Cite · Answer      │ │
│   ┌──────────────┐    │              │                             │ │
│   │ Tavily/DDG   │    │             END                            │ │
│   │ Web Search   │    └────────────────────────────────────────────┘ │
│   └──────────────┘                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd ai-orchestrator
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY (free)
```

### 3. Start services

**Terminal 1 — FastAPI backend:**
```bash
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run frontend/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker (single command)

```bash
cp .env.example .env   # fill in your keys
docker-compose up --build
```

| Service   | URL                    |
|-----------|------------------------|
| Frontend  | http://localhost:8501  |
| API       | http://localhost:8000  |
| API Docs  | http://localhost:8000/docs |

---

## 🔑 Environment Variables

### Required (minimum viable setup)

| Variable | Description | Where to get |
|----------|-------------|--------------|
| `GROQ_API_KEY` | Primary LLM provider (free) | [console.groq.com](https://console.groq.com) |

### Recommended (full feature set)

| Variable | Description | Where to get |
|----------|-------------|--------------|
| `PINECONE_API_KEY` | Vector database for RAG | [pinecone.io](https://www.pinecone.io) |
| `TAVILY_API_KEY` | Web search (1k req/month free) | [tavily.com](https://tavily.com) |
| `LANGSMITH_API_KEY` | Tracing & observability (free) | [smith.langchain.com](https://smith.langchain.com) |

### Optional

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI LLM (paid — overrides Groq) |
| `GROQ_MODEL` | Groq model name (default: `llama-3.1-8b-instant`) |
| `HF_EMBEDDING_MODEL` | HuggingFace embedding model (default: `all-MiniLM-L6-v2`) |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `ai-orchestrator`) |
| `LANGSMITH_PROJECT` | LangSmith project name (default: `ai-orchestrator`) |

---

## 🆓 Free-Tier Providers

| Component | Provider | Free Tier |
|-----------|----------|-----------|
| 🧠 LLM | [Groq](https://console.groq.com) | Generous RPM/TPM limits |
| 🧮 Embeddings | HuggingFace sentence-transformers | Unlimited (local) |
| 🗄️ Vector DB | [Pinecone Starter](https://www.pinecone.io) | 1 index · 100k vectors |
| 🔍 Web Search | [Tavily](https://tavily.com) | 1,000 searches/month |
| 🔍 Fallback Search | DuckDuckGo | Unlimited |
| 📡 Tracing | [LangSmith](https://smith.langchain.com) | 5k traces/month |
| 🖥️ Local LLM | [Ollama](https://ollama.com) | Unlimited (local) |

**No Redis. No Prometheus. No paid infrastructure.**

---

## 🧠 Agent Details

### Agent 1: Planner
- Analyses user intent with an LLM call
- Outputs structured JSON execution plan
- Determines routing: RAG-only / search-only / both / neither
- Complexity classification: low / medium / high

### Agent 2: Retrieval
- Embeds the query with HuggingFace / OpenAI
- Queries Pinecone for top-k similar chunks
- Optional LLM re-ranking of results
- Graceful fallback to in-memory keyword search

### Agent 3: Search
- Executes Tavily searches (falls back to DuckDuckGo)
- Supports multi-query parallelism
- LLM-based result filtering and relevance scoring
- Returns deduplicated, ranked snippets

### Agent 4: Synthesizer
- Merges RAG context + web search results
- Generates citation-style answer with `[Source: url]` references
- Awareness of conversation history
- Temperature tuned for accurate yet readable prose

---

## 📁 Project Structure

```
ai-orchestrator/
├── src/
│   ├── api/
│   │   └── fastapi_app.py          # REST endpoints: /chat /upload /health /metrics
│   ├── graph/
│   │   ├── graph_builder.py        # LangGraph StateGraph construction
│   │   ├── graph_state.py          # AgentState TypedDict
│   │   └── nodes.py                # Node functions + conditional routing
│   ├── agents/
│   │   ├── planner_agent.py        # Agent 1: intent analysis & planning
│   │   ├── retrieval_agent.py      # Agent 2: Pinecone RAG
│   │   ├── search_agent.py         # Agent 3: web search
│   │   └── synthesizer_agent.py    # Agent 4: answer synthesis
│   ├── llm/
│   │   ├── llm_factory.py          # Multi-provider LLM factory (Groq/OpenAI/Ollama)
│   │   └── prompts.py              # All agent prompt templates
│   ├── retrieval/
│   │   ├── pinecone_client.py      # Pinecone CRUD operations
│   │   ├── embeddings.py           # Embedding model factory
│   │   ├── ingest.py               # Document ingestion pipeline
│   │   └── retriever.py            # High-level retrieval interface
│   ├── search/
│   │   └── web_search.py           # Tavily + DuckDuckGo
│   ├── observability/
│   │   └── langsmith_tracing.py    # LangSmith utilities + WorkflowTracer
│   ├── monitoring/
│   │   └── metrics.py              # Lightweight in-process metrics
│   └── config/
│       ├── settings.py             # Pydantic settings (env-driven)
│       └── logging_config.py       # Structured logging setup
├── frontend/
│   └── streamlit_app.py            # Chat UI + file upload + metrics
├── data/uploads/                   # Uploaded document storage
├── tests/
│   └── test_workflow.py            # Pytest test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 🧪 Running Tests

```bash
pytest tests/test_workflow.py -v
```

Tests use mocks — **no API keys required** for the test suite.

---

## 📡 API Reference

### `POST /chat`
```json
{
  "query": "What is LangGraph?",
  "messages": [
    {"role": "user", "content": "Tell me about AI"},
    {"role": "assistant", "content": "AI is..."}
  ]
}
```
Response:
```json
{
  "answer": "LangGraph is a...",
  "sources": [{"title": "...", "url": "...", "type": "web"}],
  "metadata": {"plan": {...}, "rag_chunks_used": 2},
  "session_id": "uuid"
}
```

### `POST /upload`
Upload a PDF, TXT, or Markdown file via `multipart/form-data`.

### `GET /health`
Returns service status and provider configuration.

### `GET /metrics`
Returns counters, latency histograms, and error counts.

---

## 🔭 LangSmith Setup

1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Create a project named `ai-orchestrator`
3. Copy your API key to `.env`
4. All agent runs and graph executions are automatically traced

---

## 🗄️ Pinecone Setup

1. Sign up at [pinecone.io](https://www.pinecone.io) (free Starter plan)
2. Copy your API key to `.env`
3. The index (`ai-orchestrator`) is created automatically on first run
4. Upload documents via the Streamlit sidebar or `POST /upload`

---

## 🛠️ Future Improvements

- [ ] Parallel agent execution (retrieval + search simultaneously)
- [ ] Streaming responses via Server-Sent Events
- [ ] Multi-turn memory with summarisation
- [ ] Agent self-reflection / critique loop
- [ ] OpenTelemetry integration
- [ ] Kubernetes Helm chart
- [ ] Multi-namespace Pinecone support (per-user isolation)
- [ ] Hybrid BM25 + dense retrieval (Pinecone Sparse-Dense)
- [ ] Evaluation harness with RAGAS

---

## 📸 Screenshots

> Add screenshots of your Streamlit UI here after running the project.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

*Built as a portfolio-grade demonstration of production AI orchestration architecture.*
