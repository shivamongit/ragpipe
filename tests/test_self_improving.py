"""Tests for ragpipe.optimization.self_improving — Self-Improving Pipeline."""

from ragpipe.optimization.self_improving import (
    SelfImprovingLoop, FeedbackRecord, OptimizationState,
)


# ── FeedbackRecord ────────────────────────────────────────────────────────────

def test_feedback_record_defaults():
    r = FeedbackRecord(query="test", score=0.8)
    assert r.timestamp > 0
    assert r.source == "auto"


def test_feedback_record_with_params():
    r = FeedbackRecord(query="q", score=0.9, params={"top_k": 5})
    assert r.params["top_k"] == 5


# ── OptimizationState ─────────────────────────────────────────────────────────

def test_optimization_state_to_dict():
    state = OptimizationState(best_score=0.85, total_queries=10)
    d = state.to_dict()
    assert d["best_score"] == 0.85
    assert d["total_queries"] == 10


def test_optimization_state_summary():
    state = OptimizationState(best_score=0.85, total_queries=10, avg_score=0.75)
    s = state.summary()
    assert "10 queries" in s
    assert "0.85" in s


# ── SelfImprovingLoop basics ─────────────────────────────────────────────────

def test_loop_init():
    loop = SelfImprovingLoop(parameter_space={"top_k": [3, 5, 10]})
    assert loop.feedback_count == 0


def test_loop_record_feedback():
    loop = SelfImprovingLoop(parameter_space={"top_k": [3, 5, 10]})
    loop.record_feedback("What is X?", score=0.8, params={"top_k": 5})
    assert loop.feedback_count == 1
    assert loop.state.avg_score == 0.8


def test_loop_multiple_feedback():
    loop = SelfImprovingLoop(parameter_space={"top_k": [3, 5, 10]})
    loop.record_feedback("Q1", score=0.9, params={"top_k": 5})
    loop.record_feedback("Q2", score=0.7, params={"top_k": 3})
    loop.record_feedback("Q3", score=0.8, params={"top_k": 5})
    assert loop.feedback_count == 3
    assert loop.state.best_score >= 0.8  # top_k=5 averages (0.9+0.8)/2 = 0.85


def test_loop_best_params_tracked():
    loop = SelfImprovingLoop(parameter_space={"top_k": [3, 5, 10]})
    loop.record_feedback("Q1", score=0.9, params={"top_k": 5})
    loop.record_feedback("Q2", score=0.3, params={"top_k": 3})
    assert loop.state.best_params.get("top_k") == 5


# ── suggest_params ────────────────────────────────────────────────────────────

def test_suggest_params_random():
    loop = SelfImprovingLoop(
        parameter_space={"top_k": [3, 5, 10], "chunk_size": [256, 512]},
        strategy="random",
    )
    params = loop.suggest_params()
    assert "top_k" in params
    assert "chunk_size" in params
    assert params["top_k"] in [3, 5, 10]


def test_suggest_params_bandit():
    loop = SelfImprovingLoop(
        parameter_space={"top_k": [3, 5, 10]},
        strategy="bandit",
    )
    # No feedback yet → random suggestion
    params = loop.suggest_params()
    assert "top_k" in params


def test_suggest_params_bandit_with_feedback():
    loop = SelfImprovingLoop(
        parameter_space={"top_k": [3, 5, 10]},
        strategy="bandit",
    )
    loop.record_feedback("Q1", score=0.95, params={"top_k": 5})
    # With feedback, bandit should often suggest best params
    suggestions = [loop.suggest_params() for _ in range(20)]
    # At least some should be the best params
    best_count = sum(1 for s in suggestions if s.get("top_k") == 5)
    assert best_count > 0


def test_suggest_params_empty_space():
    loop = SelfImprovingLoop(parameter_space={})
    params = loop.suggest_params()
    assert params == {}


# ── set_baseline ──────────────────────────────────────────────────────────────

def test_set_baseline_improvement():
    loop = SelfImprovingLoop(parameter_space={"top_k": [3, 5]})
    loop.set_baseline(0.5)
    loop.record_feedback("Q1", score=0.75, params={"top_k": 5})
    assert loop.state.improvement_pct == 50.0


# ── optimize ──────────────────────────────────────────────────────────────────

def test_optimize_dry_run():
    loop = SelfImprovingLoop(
        parameter_space={"top_k": [3, 5, 10]},
        strategy="random",
    )
    state = loop.optimize(n_trials=5)
    assert len(state.history) == 5


def test_optimize_with_factory():
    call_count = {"n": 0}

    def factory(top_k=5):
        return {"top_k": top_k}

    def eval_fn(pipeline):
        call_count["n"] += 1
        return 0.5 + (pipeline["top_k"] / 100)

    loop = SelfImprovingLoop(
        pipeline_factory=factory,
        eval_fn=eval_fn,
        parameter_space={"top_k": [3, 5, 10]},
        strategy="random",
    )
    state = loop.optimize(n_trials=5)
    assert call_count["n"] == 5
    assert state.best_score > 0


def test_optimize_handles_factory_error():
    def bad_factory(**kwargs):
        raise RuntimeError("broken")

    def eval_fn(pipeline):
        return 0.5

    loop = SelfImprovingLoop(
        pipeline_factory=bad_factory,
        eval_fn=eval_fn,
        parameter_space={"top_k": [3, 5]},
    )
    state = loop.optimize(n_trials=3)
    assert state.total_queries == 3  # All recorded as errors


# ── Serialization ─────────────────────────────────────────────────────────────

def test_loop_to_dict():
    loop = SelfImprovingLoop(
        parameter_space={"top_k": [3, 5]},
        strategy="random",
    )
    loop.record_feedback("Q", score=0.8, params={"top_k": 5})
    d = loop.to_dict()
    assert d["strategy"] == "random"
    assert d["feedback_count"] == 1
    assert "state" in d
