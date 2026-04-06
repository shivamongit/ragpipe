"""Self-Improving Pipeline — closed-loop optimization from LLM-Judge scores.

Continuously scores query results via LLM-as-Judge + user feedback, then
uses the scores to drive automatic tuning of pipeline parameters. Supports
Bayesian optimization (via optuna if installed), online learning from
quality scores, and A/B testing between pipeline variants.

Usage:
    from ragpipe.optimization.self_improving import SelfImprovingLoop

    loop = SelfImprovingLoop(
        pipeline_factory=my_factory,
        eval_fn=my_eval,
        parameter_space={"chunk_size": [256, 512], "top_k": [3, 5, 10]},
    )
    loop.record_feedback(query="What is X?", score=0.8, metadata={})
    best = loop.optimize(n_trials=20)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FeedbackRecord:
    """A single feedback record for self-improving optimization."""
    query: str
    score: float
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    source: str = "auto"  # "auto" (LLM-Judge), "user", "metric"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class OptimizationState:
    """Current state of the self-improving loop."""
    current_params: dict[str, Any] = field(default_factory=dict)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    total_queries: int = 0
    avg_score: float = 0.0
    improvement_pct: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_params": self.current_params,
            "best_params": self.best_params,
            "best_score": round(self.best_score, 4),
            "total_queries": self.total_queries,
            "avg_score": round(self.avg_score, 4),
            "improvement_pct": round(self.improvement_pct, 2),
        }

    def summary(self) -> str:
        lines = [
            f"Self-Improving Loop: {self.total_queries} queries, "
            f"avg={self.avg_score:.4f}, best={self.best_score:.4f}",
            f"  Best params: {self.best_params}",
        ]
        if self.improvement_pct:
            lines.append(f"  Improvement: {self.improvement_pct:.1f}%")
        return "\n".join(lines)


class SelfImprovingLoop:
    """Closed-loop pipeline optimization from quality feedback.

    Records quality scores (from LLM-Judge, user feedback, or metrics),
    then optimizes pipeline parameters to maximize those scores.

    Optimization strategies:
    - "random": Random search with quality-weighted sampling
    - "bayesian": Bayesian optimization via Optuna (if installed)
    - "bandit": Multi-armed bandit (epsilon-greedy) for online learning
    """

    def __init__(
        self,
        pipeline_factory: Callable[..., Any] | None = None,
        eval_fn: Callable[..., float] | None = None,
        parameter_space: dict[str, list[Any]] | None = None,
        strategy: str = "random",
        min_samples: int = 10,
    ):
        self._factory = pipeline_factory
        self._eval_fn = eval_fn
        self._param_space = parameter_space or {}
        self._strategy = strategy
        self._min_samples = min_samples
        self._feedback: list[FeedbackRecord] = []
        self._state = OptimizationState()
        self._baseline_score: float = 0.0

    def record_feedback(
        self,
        query: str,
        score: float,
        params: dict[str, Any] | None = None,
        source: str = "auto",
        **metadata: Any,
    ) -> FeedbackRecord:
        """Record a quality feedback signal."""
        record = FeedbackRecord(
            query=query,
            score=score,
            params=params or dict(self._state.current_params),
            source=source,
            metadata=metadata,
        )
        self._feedback.append(record)
        self._update_state()
        return record

    def _update_state(self) -> None:
        """Update optimization state from recorded feedback."""
        if not self._feedback:
            return

        scores = [f.score for f in self._feedback]
        self._state.total_queries = len(self._feedback)
        self._state.avg_score = sum(scores) / len(scores)

        # Track best-performing parameter config
        param_scores: dict[str, list[float]] = {}
        for f in self._feedback:
            key = str(sorted(f.params.items()))
            if key not in param_scores:
                param_scores[key] = []
            param_scores[key].append(f.score)

        best_key = max(param_scores, key=lambda k: sum(param_scores[k]) / len(param_scores[k]))
        best_avg = sum(param_scores[best_key]) / len(param_scores[best_key])

        if best_avg > self._state.best_score:
            self._state.best_score = best_avg
            # Find the actual params for the best key
            for f in self._feedback:
                if str(sorted(f.params.items())) == best_key:
                    self._state.best_params = dict(f.params)
                    break

        if self._baseline_score > 0:
            self._state.improvement_pct = (
                (self._state.best_score - self._baseline_score) / self._baseline_score * 100
            )

    def set_baseline(self, score: float) -> None:
        """Set the baseline score for computing improvement percentage."""
        self._baseline_score = score

    def suggest_params(self) -> dict[str, Any]:
        """Suggest the next parameter configuration to try."""
        if not self._param_space:
            return dict(self._state.current_params)

        if self._strategy == "bandit":
            return self._bandit_suggest()
        elif self._strategy == "bayesian":
            return self._bayesian_suggest()
        else:
            return self._random_suggest()

    def _random_suggest(self) -> dict[str, Any]:
        """Quality-weighted random suggestion."""
        # If we have enough feedback, bias toward good configs
        if len(self._feedback) >= self._min_samples:
            good = [f for f in self._feedback if f.score >= self._state.avg_score]
            if good:
                # Sample from parameters that performed well
                ref = random.choice(good)
                params = dict(ref.params)
                # Mutate one parameter
                if self._param_space:
                    key = random.choice(list(self._param_space.keys()))
                    params[key] = random.choice(self._param_space[key])
                return params

        # Pure random
        return {k: random.choice(v) for k, v in self._param_space.items()}

    def _bandit_suggest(self, epsilon: float = 0.2) -> dict[str, Any]:
        """Epsilon-greedy multi-armed bandit suggestion."""
        if random.random() < epsilon or not self._state.best_params:
            return self._random_suggest()
        return dict(self._state.best_params)

    def _bayesian_suggest(self) -> dict[str, Any]:
        """Bayesian optimization suggestion (Optuna if available)."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = {}
                for key, values in self._param_space.items():
                    if all(isinstance(v, int) for v in values):
                        params[key] = trial.suggest_int(key, min(values), max(values))
                    elif all(isinstance(v, float) for v in values):
                        params[key] = trial.suggest_float(key, min(values), max(values))
                    else:
                        params[key] = trial.suggest_categorical(key, values)

                if self._factory and self._eval_fn:
                    pipeline = self._factory(**params)
                    return self._eval_fn(pipeline)

                # Use historical data
                matching = [
                    f.score for f in self._feedback
                    if all(f.params.get(k) == v for k, v in params.items())
                ]
                return sum(matching) / len(matching) if matching else 0.0

            study = optuna.create_study(direction="maximize")

            # Seed with existing observations
            for f in self._feedback:
                try:
                    distributions = {}
                    for key, values in self._param_space.items():
                        if all(isinstance(v, int) for v in values):
                            distributions[key] = optuna.distributions.IntDistribution(min(values), max(values))
                        elif all(isinstance(v, float) for v in values):
                            distributions[key] = optuna.distributions.FloatDistribution(min(values), max(values))
                        else:
                            distributions[key] = optuna.distributions.CategoricalDistribution(values)

                    trial = optuna.trial.create_trial(
                        params={k: f.params.get(k, values[0]) for k, values in self._param_space.items()},
                        distributions=distributions,
                        values=[f.score],
                    )
                    study.add_trial(trial)
                except Exception:
                    pass

            study.optimize(objective, n_trials=1)
            return study.best_params

        except ImportError:
            return self._random_suggest()

    def optimize(
        self,
        n_trials: int = 20,
        queries: list[str] | None = None,
    ) -> OptimizationState:
        """Run optimization loop for n_trials.

        If pipeline_factory and eval_fn are provided, runs actual trials.
        Otherwise, suggests parameters based on recorded feedback.
        """
        for i in range(n_trials):
            params = self.suggest_params()
            self._state.current_params = params

            if self._factory and self._eval_fn:
                try:
                    pipeline = self._factory(**params)
                    score = self._eval_fn(pipeline)
                    self.record_feedback(
                        query=f"trial_{i}",
                        score=score,
                        params=params,
                        source="optimization",
                    )
                except Exception:
                    self.record_feedback(
                        query=f"trial_{i}",
                        score=0.0,
                        params=params,
                        source="optimization_error",
                    )
            else:
                # Dry run — just record the suggestion
                self._state.history.append({
                    "trial": i,
                    "params": params,
                    "timestamp": time.time(),
                })

        return self._state

    @property
    def feedback_count(self) -> int:
        return len(self._feedback)

    @property
    def state(self) -> OptimizationState:
        return self._state

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self._strategy,
            "feedback_count": self.feedback_count,
            "parameter_space": self._param_space,
            "state": self._state.to_dict(),
        }
