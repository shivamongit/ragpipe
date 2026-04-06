"""Tests for Pipeline Optimizer."""

from ragpipe.optimization.optimizer import (
    PipelineOptimizer, OptimizationResult, ParameterSpace, Trial,
)


def test_parameter_space_grid_size():
    space = ParameterSpace(chunk_size=[256, 512], top_k=[3, 5, 10])
    assert space.grid_size == 6  # 2 * 3


def test_parameter_space_grid_configs():
    space = ParameterSpace(chunk_size=[256, 512], top_k=[3, 5])
    configs = space.grid_configs()
    assert len(configs) == 4
    assert {"chunk_size": 256, "top_k": 3} in configs
    assert {"chunk_size": 512, "top_k": 5} in configs


def test_parameter_space_random_configs():
    space = ParameterSpace(chunk_size=[256, 512, 1024], top_k=[3, 5, 10])
    configs = space.random_configs(5)
    assert len(configs) == 5
    for c in configs:
        assert c["chunk_size"] in [256, 512, 1024]
        assert c["top_k"] in [3, 5, 10]


def test_parameter_space_empty():
    space = ParameterSpace()
    assert space.grid_size == 0
    assert space.grid_configs() == [{}]


def test_optimizer_grid_search():
    """Optimizer should find best params via grid search."""
    # Simulate: score = chunk_size/1024 + top_k/10
    def build_pipeline(chunk_size=512, top_k=5):
        return {"chunk_size": chunk_size, "top_k": top_k}

    def eval_fn(pipeline, dataset):
        return pipeline["chunk_size"] / 1024 + pipeline["top_k"] / 10

    optimizer = PipelineOptimizer(
        pipeline_factory=build_pipeline,
        eval_fn=eval_fn,
        eval_dataset=None,
    )

    space = ParameterSpace(chunk_size=[256, 512, 1024], top_k=[3, 5, 10])
    result = optimizer.optimize(space, method="grid")

    assert result.best_params == {"chunk_size": 1024, "top_k": 10}
    assert result.best_score == 1024 / 1024 + 10 / 10  # 2.0
    assert len(result.trials) == 9  # 3 * 3
    assert result.method == "grid"


def test_optimizer_random_search():
    def build_pipeline(x=1):
        return {"x": x}

    def eval_fn(pipeline, dataset):
        return pipeline["x"]

    optimizer = PipelineOptimizer(
        pipeline_factory=build_pipeline,
        eval_fn=eval_fn,
    )
    space = ParameterSpace(x=[1, 2, 3, 4, 5])
    result = optimizer.optimize(space, method="random", n_random=3)
    assert len(result.trials) == 3
    assert result.best_score >= 1


def test_optimizer_handles_errors():
    """Optimizer should handle trial errors gracefully."""
    def build_pipeline(x=1):
        if x == 2:
            raise ValueError("bad config")
        return {"x": x}

    def eval_fn(pipeline, dataset):
        return pipeline["x"]

    optimizer = PipelineOptimizer(
        pipeline_factory=build_pipeline,
        eval_fn=eval_fn,
    )
    space = ParameterSpace(x=[1, 2, 3])
    result = optimizer.optimize(space, method="grid")

    assert result.best_score == 3.0
    assert result.best_params == {"x": 3}
    # One trial should have an error
    errors = [t for t in result.trials if t.error is not None]
    assert len(errors) == 1


def test_optimization_result_summary():
    result = OptimizationResult(
        best_params={"k": 5},
        best_score=0.9,
        trials=[
            Trial(params={"k": 3}, score=0.7, duration_ms=10),
            Trial(params={"k": 5}, score=0.9, duration_ms=12),
        ],
        total_duration_ms=22,
        method="grid",
        search_space_size=2,
    )
    summary = result.summary()
    assert "0.9" in summary
    assert "grid" in summary


def test_optimization_result_to_dict():
    result = OptimizationResult(
        best_params={"k": 5},
        best_score=0.9,
        trials=[Trial(params={"k": 5}, score=0.9, duration_ms=10)],
        total_duration_ms=10,
        method="grid",
        search_space_size=1,
    )
    d = result.to_dict()
    assert d["best_score"] == 0.9
    assert len(d["trials"]) == 1


def test_optimization_result_metrics():
    result = OptimizationResult(
        best_params={"k": 5},
        best_score=0.9,
        trials=[
            Trial(params={"k": 3}, score=0.5, duration_ms=10),
            Trial(params={"k": 5}, score=0.9, duration_ms=12),
        ],
        total_duration_ms=22,
        method="grid",
        search_space_size=2,
    )
    assert result.worst_score == 0.5
    assert result.mean_score == 0.7
    assert result.improvement > 0


def test_optimizer_single_param():
    def build(a=1):
        return {"a": a}

    def evaluate(p, d):
        return p["a"] * 10

    opt = PipelineOptimizer(pipeline_factory=build, eval_fn=evaluate)
    result = opt.optimize(ParameterSpace(a=[1]), method="grid")
    assert result.best_score == 10
    assert len(result.trials) == 1
