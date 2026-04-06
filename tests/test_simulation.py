"""Tests for ragpipe.simulation — Retrieval Simulation Environment."""

from dataclasses import dataclass
from ragpipe.simulation.runner import (
    SimulationRunner, SimulationResult, FailureScenario, QueryResult,
)


@dataclass
class _MockResult:
    score: float
    class chunk:
        text = "mock"
        id = "mock:0"


def _mock_retrieve(query):
    if not query or not query.strip():
        return []
    return [_MockResult(score=0.8), _MockResult(score=0.6)]


# ── QueryResult ───────────────────────────────────────────────────────────────

def test_query_result_to_dict():
    qr = QueryResult(query="test", scenario="custom", passed=True, top_score=0.8)
    d = qr.to_dict()
    assert d["query"] == "test"
    assert d["passed"] is True


# ── SimulationResult ──────────────────────────────────────────────────────────

def test_simulation_result_pass_rate():
    r = SimulationResult(total_queries=10, passed=8, failed=2)
    assert r.pass_rate == 0.8


def test_simulation_result_pass_rate_empty():
    r = SimulationResult()
    assert r.pass_rate == 0.0


def test_simulation_result_summary():
    r = SimulationResult(
        total_queries=5, passed=4, failed=1, avg_latency_ms=10.0,
        scenarios_tested=["adversarial_queries"], duration_s=0.5,
    )
    s = r.summary()
    assert "4/5" in s
    assert "80%" in s


def test_simulation_result_to_dict():
    r = SimulationResult(total_queries=1, passed=1)
    d = r.to_dict()
    assert d["total_queries"] == 1
    assert "pass_rate" in d


# ── SimulationRunner basics ───────────────────────────────────────────────────

def test_runner_no_pipeline():
    runner = SimulationRunner()
    result = runner.run(queries=["test"])
    assert result.total_queries == 1
    assert result.failed == 1  # No pipeline provided


def test_runner_with_retrieve_fn():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    result = runner.run(queries=["What is RAG?"])
    assert result.total_queries == 1
    assert result.passed == 1


def test_runner_custom_queries():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_queries(["Q1", "Q2"], scenario="custom")
    result = runner.run()
    assert result.total_queries == 2


# ── Scenarios ─────────────────────────────────────────────────────────────────

def test_runner_adversarial():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.ADVERSARIAL_QUERIES)
    result = runner.run()
    assert result.total_queries > 0
    assert "adversarial_queries" in result.scenarios_tested


def test_runner_missing_context():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.MISSING_CONTEXT)
    result = runner.run()
    assert result.total_queries >= 3


def test_runner_ambiguous():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.AMBIGUOUS_QUERIES)
    result = runner.run()
    assert result.total_queries >= 3


def test_runner_empty_retrieval():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.EMPTY_RETRIEVAL)
    result = runner.run()
    assert result.total_queries >= 2


def test_runner_long_queries():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.LONG_QUERIES)
    result = runner.run()
    assert result.total_queries >= 1


def test_runner_low_relevance():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.LOW_RELEVANCE)
    result = runner.run()
    assert result.total_queries >= 1


def test_runner_stale_data():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.STALE_DATA)
    result = runner.run()
    assert result.total_queries >= 1


def test_runner_embedding_noise():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.EMBEDDING_NOISE)
    result = runner.run()
    assert result.total_queries >= 1


# ── Multiple scenarios ────────────────────────────────────────────────────────

def test_runner_multiple_scenarios():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_scenario(FailureScenario.ADVERSARIAL_QUERIES)
    runner.add_scenario(FailureScenario.MISSING_CONTEXT)
    result = runner.run()
    assert result.total_queries >= 5
    assert len(result.scenarios_tested) >= 2


# ── Custom assertions ─────────────────────────────────────────────────────────

def test_runner_custom_assertion_pass():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_assertion(lambda r: r.retrieved_count > 0)
    result = runner.run(queries=["What is X?"])
    assert result.passed == 1


def test_runner_custom_assertion_fail():
    runner = SimulationRunner(retrieve_fn=_mock_retrieve)
    runner.add_assertion(lambda r: r.top_score > 0.99)  # Impossible threshold
    result = runner.run(queries=["What is X?"])
    assert result.failed == 1


# ── Latency threshold ─────────────────────────────────────────────────────────

def test_runner_latency_threshold():
    import time

    def slow_retrieve(q):
        time.sleep(0.01)
        return [_MockResult(score=0.5)]

    runner = SimulationRunner(
        retrieve_fn=slow_retrieve,
        max_latency_ms=5.0,  # Very tight threshold
    )
    result = runner.run(queries=["test"])
    assert result.failed == 1
    assert "Latency" in result.query_results[0].failure_reason


# ── Exception handling ────────────────────────────────────────────────────────

def test_runner_handles_exceptions():
    def broken_retrieve(q):
        raise RuntimeError("Retrieval failed!")

    runner = SimulationRunner(retrieve_fn=broken_retrieve)
    result = runner.run(queries=["test"])
    assert result.failed == 1
    assert "RuntimeError" in result.query_results[0].failure_reason
