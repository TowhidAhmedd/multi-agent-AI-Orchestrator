"""
Test suite for the Multi-Agent AI Orchestrator.

Tests are designed to run without real API keys using mocks,
so CI can validate logic without cloud dependencies.

Run:
    pytest tests/test_workflow.py -v
"""

import json
import os
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_state() -> Dict[str, Any]:
    from src.graph.graph_state import create_initial_state
    return create_initial_state(query="What is LangGraph?")


@pytest.fixture()
def planner_output_state(sample_state) -> Dict[str, Any]:
    return {
        **sample_state,
        "execution_plan": {
            "needs_search": True,
            "needs_rag": True,
            "complexity": "medium",
            "search_queries": ["LangGraph overview"],
            "rag_query": "What is LangGraph?",
            "reasoning": "Mixed query needing both RAG and web search.",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — graph state
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphState:
    def test_initial_state_has_required_keys(self, sample_state):
        required = {"query", "execution_plan", "rag_context", "search_results",
                    "final_answer", "sources", "messages", "file_paths"}
        assert required.issubset(set(sample_state.keys()))

    def test_initial_state_query(self, sample_state):
        assert sample_state["query"] == "What is LangGraph?"

    def test_initial_state_empty_collections(self, sample_state):
        assert sample_state["rag_context"] == []
        assert sample_state["search_results"] == []
        assert sample_state["sources"] == []

    def test_initial_state_no_error(self, sample_state):
        assert sample_state["error"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — planner agent
# ─────────────────────────────────────────────────────────────────────────────

class TestPlannerAgent:
    @patch("src.agents.planner_agent.get_llm")
    def test_planner_returns_valid_plan(self, mock_llm_factory, sample_state):
        mock_chain_output = json.dumps({
            "needs_search": True,
            "needs_rag": False,
            "complexity": "low",
            "search_queries": ["LangGraph"],
            "rag_query": "What is LangGraph?",
            "reasoning": "Simple factual query.",
        })
        mock_llm = MagicMock()
        mock_llm.__or__ = lambda self, other: MagicMock(
            invoke=lambda _: mock_chain_output
        )
        mock_llm_factory.return_value = mock_llm

        from src.agents.planner_agent import _parse_plan
        plan = _parse_plan(mock_chain_output)

        assert plan["needs_search"] is True
        assert plan["complexity"] == "low"
        assert "LangGraph" in plan["search_queries"]

    def test_parse_plan_handles_json_in_markdown(self):
        from src.agents.planner_agent import _parse_plan

        raw = "```json\n{\"needs_search\": true, \"needs_rag\": false, \"complexity\": \"low\", \"search_queries\": [], \"rag_query\": \"\", \"reasoning\": \"test\"}\n```"
        plan = _parse_plan(raw)
        assert plan["needs_search"] is True
        assert plan["complexity"] == "low"

    def test_parse_plan_fallback_on_invalid_json(self):
        from src.agents.planner_agent import _parse_plan, _DEFAULT_PLAN

        plan = _parse_plan("This is not JSON at all!")
        assert plan["needs_search"] == _DEFAULT_PLAN["needs_search"]


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — routing logic
# ─────────────────────────────────────────────────────────────────────────────

class TestRouting:
    def test_route_both_rag_and_search(self, planner_output_state):
        from src.graph.nodes import route_after_planner
        route = route_after_planner(planner_output_state)
        assert route == "retrieval"  # RAG takes priority

    def test_route_search_only(self, sample_state):
        from src.graph.nodes import route_after_planner
        state = {
            **sample_state,
            "execution_plan": {
                "needs_rag": False,
                "needs_search": True,
                "search_queries": ["test"],
            },
        }
        route = route_after_planner(state)
        assert route == "search"

    def test_route_no_external_sources(self, sample_state):
        from src.graph.nodes import route_after_planner
        state = {
            **sample_state,
            "execution_plan": {"needs_rag": False, "needs_search": False},
        }
        route = route_after_planner(state)
        assert route == "synthesizer"

    def test_route_after_retrieval_with_search(self, planner_output_state):
        from src.graph.nodes import route_after_retrieval
        route = route_after_retrieval(planner_output_state)
        assert route == "search"

    def test_route_after_retrieval_no_search(self, sample_state):
        from src.graph.nodes import route_after_retrieval
        state = {
            **sample_state,
            "execution_plan": {"needs_rag": True, "needs_search": False},
        }
        route = route_after_retrieval(state)
        assert route == "synthesizer"


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — retriever
# ─────────────────────────────────────────────────────────────────────────────

class TestRetriever:
    def test_local_keyword_search(self):
        from src.retrieval.retriever import _keyword_search, _local_store

        _local_store.clear()
        _local_store.append(
            {"metadata": {"text": "LangGraph is a graph-based orchestration framework."}}
        )
        results = _keyword_search("LangGraph", top_k=3)
        assert len(results) == 1

    def test_retrieve_returns_empty_without_store(self):
        from src.retrieval import retriever
        retriever._local_store.clear()

        with patch("src.retrieval.retriever.settings") as mock_settings:
            mock_settings.pinecone_api_key = ""
            mock_settings.top_k_results = 5
            results = retriever.retrieve("test query")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — web search
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSearch:
    def test_format_search_results_empty(self):
        from src.search.web_search import format_search_results
        result = format_search_results([])
        assert "No search results" in result

    def test_format_search_results(self):
        from src.search.web_search import format_search_results
        results = [{"title": "Test", "url": "https://example.com", "snippet": "A test snippet."}]
        formatted = format_search_results(results)
        assert "Test" in formatted
        assert "example.com" in formatted

    @patch("src.search.web_search._ddg_search")
    def test_web_search_uses_ddg_when_no_tavily(self, mock_ddg):
        mock_ddg.return_value = [{"title": "DDG result", "url": "https://ddg.com", "snippet": "test"}]

        with patch("src.search.web_search.settings") as mock_settings:
            mock_settings.tavily_api_key = ""
            from src.search.web_search import web_search
            results = web_search("test query")

        mock_ddg.assert_called_once()
        assert len(results) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def setup_method(self):
        from src.monitoring.metrics import reset
        reset()

    def test_increment(self):
        from src.monitoring import metrics
        metrics.increment("test.counter")
        metrics.increment("test.counter")
        snap = metrics.get_snapshot()
        assert snap["counters"]["test.counter"] == 2

    def test_record_latency(self):
        from src.monitoring import metrics
        metrics.record_latency("test.op", 0.5)
        metrics.record_latency("test.op", 1.0)
        snap = metrics.get_snapshot()
        assert snap["latency"]["test.op"]["count"] == 2
        assert snap["latency"]["test.op"]["mean_ms"] == pytest.approx(750.0, rel=0.01)

    def test_timer_context_manager(self):
        from src.monitoring.metrics import Timer, get_snapshot
        import time
        with Timer("test.timer"):
            time.sleep(0.05)
        snap = get_snapshot()
        assert "test.timer" in snap["latency"]
        assert snap["latency"]["test.timer"]["count"] == 1

    def test_record_error(self):
        from src.monitoring import metrics
        metrics.record_error("test.component")
        snap = metrics.get_snapshot()
        assert snap["errors"]["test.component"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — full graph build (no LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphBuild:
    def test_graph_compiles(self):
        """Verify that the LangGraph StateGraph compiles without errors."""
        from src.graph.graph_builder import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        from src.graph.graph_builder import build_graph
        graph = build_graph()
        # LangGraph compiled graph exposes .nodes
        node_names = set(graph.nodes.keys())
        expected = {"planner", "retrieval", "search", "synthesizer"}
        assert expected.issubset(node_names)
