"""Tests for ragpipe.agents.planner — Agentic Retrieval System."""

from dataclasses import dataclass, field
from ragpipe.agents.planner import (
    RetrievalPlanner, RetrievalEvaluator, AgenticPipeline,
    AgenticResult, RetrievalPlan, PlanStep, PlanStepType,
)


@dataclass
class _MockRetrievalResult:
    score: float
    class chunk:
        text = "mock chunk"
        id = "mock:0"
        metadata = {}


def _mock_retrieve(query):
    if "nonexistent" in query.lower():
        return []
    return [_MockRetrievalResult(score=0.8), _MockRetrievalResult(score=0.6)]


def _mock_generate(query, results):
    return f"Answer to: {query}"


# ── PlanStep ──────────────────────────────────────────────────────────────────

def test_plan_step_to_dict():
    step = PlanStep(step_id=1, query="test", step_type=PlanStepType.SEARCH)
    d = step.to_dict()
    assert d["step_id"] == 1
    assert d["step_type"] == "search"


# ── RetrievalPlan ─────────────────────────────────────────────────────────────

def test_retrieval_plan_to_dict():
    plan = RetrievalPlan(
        original_query="test",
        steps=[PlanStep(step_id=1, query="sub")],
        reasoning="simple",
    )
    d = plan.to_dict()
    assert d["original_query"] == "test"
    assert len(d["steps"]) == 1


# ── AgenticResult ─────────────────────────────────────────────────────────────

def test_agentic_result_to_dict():
    r = AgenticResult(answer="test", confidence=0.85, retrieval_rounds=2)
    d = r.to_dict()
    assert d["answer"] == "test"
    assert d["confidence"] == 0.85


def test_agentic_result_summary():
    r = AgenticResult(
        answer="The answer is 42",
        confidence=0.9,
        retrieval_rounds=2,
        total_chunks_retrieved=5,
    )
    s = r.summary()
    assert "confidence=0.90" in s
    assert "rounds=2" in s


# ── RetrievalPlanner heuristic ────────────────────────────────────────────────

def test_planner_simple_query():
    planner = RetrievalPlanner()
    plan = planner.plan("What is machine learning?")
    assert len(plan.steps) == 1
    assert plan.steps[0].step_type == PlanStepType.SEARCH


def test_planner_comparison_query():
    planner = RetrievalPlanner()
    plan = planner.plan("Compare Python and Java for web development")
    assert len(plan.steps) >= 2
    # Should have a COMPARE step
    types = [s.step_type for s in plan.steps]
    assert PlanStepType.COMPARE in types


def test_planner_comparison_vs():
    planner = RetrievalPlanner()
    plan = planner.plan("React vs Vue for frontend")
    assert len(plan.steps) >= 2


def test_planner_multi_hop():
    planner = RetrievalPlanner()
    plan = planner.plan("Find the CEO of Acme Corp then look up their education")
    assert plan.estimated_hops >= 2


def test_planner_aggregation():
    planner = RetrievalPlanner()
    plan = planner.plan("How many employees does Acme have?")
    types = [s.step_type for s in plan.steps]
    assert PlanStepType.AGGREGATE in types


def test_planner_list_all():
    planner = RetrievalPlanner()
    plan = planner.plan("List all products in the catalog")
    types = [s.step_type for s in plan.steps]
    assert PlanStepType.AGGREGATE in types


# ── RetrievalPlanner with LLM ────────────────────────────────────────────────

def test_planner_with_llm():
    def mock_plan_fn(prompt):
        return '{"steps": [{"query": "step 1", "type": "search"}, {"query": "step 2", "type": "verify"}], "reasoning": "two steps"}'

    planner = RetrievalPlanner(plan_fn=mock_plan_fn)
    plan = planner.plan("Complex query")
    assert len(plan.steps) == 2
    assert plan.reasoning == "two steps"


def test_planner_llm_fallback_on_bad_json():
    def bad_plan_fn(prompt):
        return "not valid json"

    planner = RetrievalPlanner(plan_fn=bad_plan_fn)
    plan = planner.plan("Simple query")
    assert len(plan.steps) >= 1  # Falls back to heuristic


# ── RetrievalEvaluator ───────────────────────────────────────────────────────

def test_evaluator_good_results():
    evaluator = RetrievalEvaluator(min_score=0.3, min_results=2)
    results = [_MockRetrievalResult(score=0.8), _MockRetrievalResult(score=0.6)]
    eval_result = evaluator.evaluate("test", results)
    assert eval_result["quality"] == "good"
    assert eval_result["needs_more"] is False


def test_evaluator_empty_results():
    evaluator = RetrievalEvaluator()
    eval_result = evaluator.evaluate("test", [])
    assert eval_result["quality"] == "insufficient"
    assert eval_result["needs_more"] is True


def test_evaluator_low_score():
    evaluator = RetrievalEvaluator(min_score=0.9)
    results = [_MockRetrievalResult(score=0.3)]
    eval_result = evaluator.evaluate("test", results, round_num=1)
    assert eval_result["quality"] == "poor"


def test_evaluator_no_retry_after_max_rounds():
    evaluator = RetrievalEvaluator(min_score=0.9)
    results = [_MockRetrievalResult(score=0.3)]
    eval_result = evaluator.evaluate("test", results, round_num=3)
    assert eval_result["needs_more"] is False


# ── AgenticPipeline ───────────────────────────────────────────────────────────

def test_agentic_simple_query():
    agent = AgenticPipeline(
        retrieve_fn=_mock_retrieve,
        generate_fn=_mock_generate,
    )
    result = agent.run("What is RAG?")
    assert result.answer != ""
    assert result.retrieval_rounds >= 1
    assert result.total_chunks_retrieved >= 1


def test_agentic_comparison_query():
    agent = AgenticPipeline(
        retrieve_fn=_mock_retrieve,
        generate_fn=_mock_generate,
    )
    result = agent.run("Compare FAISS and ChromaDB")
    assert result.plan is not None
    assert len(result.plan.steps) >= 2
    assert result.retrieval_rounds >= 2


def test_agentic_no_results():
    def empty_retrieve(q):
        return []

    agent = AgenticPipeline(
        retrieve_fn=empty_retrieve,
        generate_fn=_mock_generate,
    )
    result = agent.run("What about nonexistent things?")
    assert "could not find" in result.answer.lower() or result.answer != ""


def test_agentic_with_critique():
    agent = AgenticPipeline(
        retrieve_fn=_mock_retrieve,
        generate_fn=_mock_generate,
        critique_fn=lambda q, a, r: "Looks good but could use more sources",
    )
    result = agent.run("What is X?")
    assert "more sources" in result.critique


def test_agentic_confidence():
    agent = AgenticPipeline(
        retrieve_fn=_mock_retrieve,
        generate_fn=_mock_generate,
    )
    result = agent.run("What is RAG?")
    assert 0.0 <= result.confidence <= 1.0


def test_agentic_latency_tracked():
    agent = AgenticPipeline(
        retrieve_fn=_mock_retrieve,
        generate_fn=_mock_generate,
    )
    result = agent.run("Quick question")
    assert result.latency_ms > 0
