"""Pipeline Optimizer — auto-tune RAG parameters from evaluation data.

Inspired by DSPy's compile-time optimization, but simpler and built-in.
No other RAG framework (LangChain, LlamaIndex, Haystack) provides built-in
parameter optimization. DSPy does prompt optimization but not RAG-specific
parameter tuning (chunk_size, top_k, overlap, similarity thresholds).

The optimizer:
1. Takes a pipeline factory function + evaluation dataset
2. Defines a parameter search space (chunk_size, top_k, overlap, etc.)
3. Runs grid search or random search over the space
4. Evaluates each configuration using your metric function
5. Returns the best configuration with full trial history

Usage:
    from ragpipe.optimization import PipelineOptimizer, ParameterSpace

    space = ParameterSpace(
        chunk_size=[256, 512, 1024],
        top_k=[3, 5, 10],
        overlap=[32, 64, 128],
    )

    optimizer = PipelineOptimizer(
        pipeline_factory=build_pipeline,   # fn(params) -> Pipeline
        eval_fn=evaluate_pipeline,         # fn(pipeline, dataset) -> float
        eval_dataset=my_qa_pairs,
    )

    result = optimizer.optimize(space, method="grid")
    print(result.best_params)    # {"chunk_size": 512, "top_k": 5, "overlap": 64}
    print(result.best_score)     # 0.87
    print(result.trials)         # full history of all trials
"""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ParameterSpace:
    """Defines the search space for pipeline optimization.

    Each parameter maps to a list of candidate values to try.
    """
    params: dict[str, list[Any]] = field(default_factory=dict)

    def __init__(self, **kwargs: list[Any]):
        self.params = {k: v for k, v in kwargs.items() if isinstance(v, list) and v}

    @property
    def grid_size(self) -> int:
        """Total number of configurations in grid search."""
        if not self.params:
            return 0
        size = 1
        for values in self.params.values():
            size *= len(values)
        return size

    def grid_configs(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        if not self.params:
            return [{}]
        keys = list(self.params.keys())
        values = list(self.params.values())
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs

    def random_configs(self, n: int) -> list[dict[str, Any]]:
        """Generate n random parameter combinations."""
        if not self.params:
            return [{}]
        configs = []
        for _ in range(n):
            config = {}
            for key, values in self.params.items():
                config[key] = random.choice(values)
            configs.append(config)
        return configs


@dataclass
class Trial:
    """A single optimization trial."""
    params: dict[str, Any]
    score: float
    duration_ms: float
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of pipeline optimization."""
    best_params: dict[str, Any]
    best_score: float
    trials: list[Trial]
    total_duration_ms: float
    method: str
    search_space_size: int

    @property
    def worst_score(self) -> float:
        valid = [t for t in self.trials if t.error is None]
        return min(t.score for t in valid) if valid else 0.0

    @property
    def mean_score(self) -> float:
        valid = [t for t in self.trials if t.error is None]
        return sum(t.score for t in valid) / len(valid) if valid else 0.0

    @property
    def improvement(self) -> float:
        """Improvement from worst to best."""
        if self.worst_score == 0:
            return 0.0
        return (self.best_score - self.worst_score) / max(self.worst_score, 1e-9)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Optimization Result ({self.method} search)",
            f"  Search space: {self.search_space_size} configurations",
            f"  Trials run:   {len(self.trials)}",
            f"  Best score:   {self.best_score:.4f}",
            f"  Mean score:   {self.mean_score:.4f}",
            f"  Worst score:  {self.worst_score:.4f}",
            f"  Improvement:  {self.improvement:.1%}",
            f"  Duration:     {self.total_duration_ms:.0f}ms",
            f"  Best params:  {self.best_params}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "mean_score": self.mean_score,
            "worst_score": self.worst_score,
            "improvement": self.improvement,
            "method": self.method,
            "search_space_size": self.search_space_size,
            "total_duration_ms": self.total_duration_ms,
            "trials": [
                {"params": t.params, "score": t.score, "duration_ms": t.duration_ms, "error": t.error}
                for t in self.trials
            ],
        }


class PipelineOptimizer:
    """Auto-tune RAG pipeline parameters by evaluating against a dataset.

    This is a unique ragpipe feature — no other framework provides built-in
    RAG parameter optimization. DSPy optimizes prompts; this optimizes the
    infrastructure parameters (chunk_size, top_k, overlap, thresholds) that
    have massive impact on retrieval quality.
    """

    def __init__(
        self,
        pipeline_factory: Callable[..., Any],
        eval_fn: Callable[..., float],
        eval_dataset: Any = None,
        verbose: bool = False,
    ):
        self.pipeline_factory = pipeline_factory
        self.eval_fn = eval_fn
        self.eval_dataset = eval_dataset
        self.verbose = verbose

    def _run_trial(self, params: dict[str, Any]) -> Trial:
        """Run a single trial with given parameters."""
        start = time.perf_counter()
        try:
            pipeline = self.pipeline_factory(**params)
            score = self.eval_fn(pipeline, self.eval_dataset)
            duration_ms = (time.perf_counter() - start) * 1000
            if self.verbose:
                print(f"  Trial {params} → {score:.4f} ({duration_ms:.0f}ms)")
            return Trial(params=params, score=float(score), duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            if self.verbose:
                print(f"  Trial {params} → ERROR: {e}")
            return Trial(params=params, score=0.0, duration_ms=duration_ms, error=str(e))

    def optimize(
        self,
        space: ParameterSpace,
        method: str = "grid",
        n_random: int = 20,
    ) -> OptimizationResult:
        """Run optimization over the parameter space.

        Args:
            space: Parameter search space
            method: "grid" for exhaustive search, "random" for random sampling
            n_random: Number of random samples (only used if method="random")

        Returns:
            OptimizationResult with best parameters and full trial history
        """
        start = time.perf_counter()

        if method == "grid":
            configs = space.grid_configs()
        elif method == "random":
            configs = space.random_configs(n_random)
        else:
            raise ValueError(f"Unknown method {method!r}. Use 'grid' or 'random'.")

        if self.verbose:
            print(f"Running {method} search over {len(configs)} configurations...")

        trials = [self._run_trial(config) for config in configs]

        valid_trials = [t for t in trials if t.error is None]
        if valid_trials:
            best = max(valid_trials, key=lambda t: t.score)
            best_params = best.params
            best_score = best.score
        else:
            best_params = configs[0] if configs else {}
            best_score = 0.0

        total_ms = (time.perf_counter() - start) * 1000

        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            trials=trials,
            total_duration_ms=total_ms,
            method=method,
            search_space_size=space.grid_size,
        )

        if self.verbose:
            print(result.summary())

        return result

    async def aoptimize(
        self,
        space: ParameterSpace,
        method: str = "grid",
        n_random: int = 20,
    ) -> OptimizationResult:
        """Async version of optimize."""
        return await asyncio.to_thread(self.optimize, space, method, n_random)
