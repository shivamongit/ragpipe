"""Tests for ragpipe.utils.costs — cost tracking."""

from ragpipe.utils.costs import CostTracker, UsageRecord


def test_cost_tracker_defaults():
    tracker = CostTracker()
    assert tracker.total_cost == 0.0
    assert tracker.total_tokens == 0
    assert tracker.query_count == 0


def test_record_generation():
    tracker = CostTracker()
    record = tracker.record_generation("gpt-4o", prompt_tokens=500, completion_tokens=200)
    assert record.model == "gpt-4o"
    assert record.operation == "generation"
    assert record.total_tokens == 700
    assert record.cost_usd > 0


def test_record_embedding():
    tracker = CostTracker()
    record = tracker.record_embedding("text-embedding-3-small", token_count=10000)
    assert record.model == "text-embedding-3-small"
    assert record.total_tokens == 10000
    assert record.cost_usd > 0


def test_local_model_free():
    tracker = CostTracker()
    tracker.record_generation("ollama/gemma4", prompt_tokens=1000, completion_tokens=500)
    assert tracker.total_cost == 0.0


def test_cost_by_model():
    tracker = CostTracker()
    tracker.record_generation("gpt-4o", prompt_tokens=100, completion_tokens=50)
    tracker.record_generation("gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    by_model = tracker.cost_by_model()
    assert "gpt-4o" in by_model
    assert "gpt-4o-mini" in by_model


def test_cost_by_operation():
    tracker = CostTracker()
    tracker.record_generation("gpt-4o", prompt_tokens=100, completion_tokens=50)
    tracker.record_embedding("text-embedding-3-small", token_count=1000)
    by_op = tracker.cost_by_operation()
    assert "generation" in by_op
    assert "embedding" in by_op


def test_budget_enforcement():
    tracker = CostTracker(budget_usd=0.001)
    tracker.record_generation("gpt-4o", prompt_tokens=10000, completion_tokens=5000)
    assert tracker.is_over_budget is True
    assert tracker.remaining_budget == 0.0


def test_budget_not_set():
    tracker = CostTracker()
    assert tracker.is_over_budget is False
    assert tracker.remaining_budget is None


def test_summary():
    tracker = CostTracker()
    tracker.record_generation("gpt-4o", prompt_tokens=100, completion_tokens=50)
    summary = tracker.summary()
    assert "gpt-4o" in summary
    assert "1 operations" in summary


def test_to_dict():
    tracker = CostTracker()
    tracker.record_generation("gpt-4o", prompt_tokens=100, completion_tokens=50)
    d = tracker.to_dict()
    assert "total_cost_usd" in d
    assert "records" in d
    assert len(d["records"]) == 1


def test_clear():
    tracker = CostTracker()
    tracker.record_generation("gpt-4o", prompt_tokens=100, completion_tokens=50)
    tracker.clear()
    assert tracker.query_count == 0
    assert tracker.total_cost == 0.0


def test_tokens_by_model():
    tracker = CostTracker()
    tracker.record_generation("gpt-4o", prompt_tokens=100, completion_tokens=50)
    tracker.record_generation("gpt-4o", prompt_tokens=200, completion_tokens=100)
    by_model = tracker.tokens_by_model()
    assert by_model["gpt-4o"] == 450
