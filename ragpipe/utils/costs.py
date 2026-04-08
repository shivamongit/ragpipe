"""Cost tracking for LLM API usage — per-query, per-model, and cumulative.

Tracks token usage and computes estimated costs based on configurable
per-model pricing. Supports both embedding and generation cost tracking.

Usage:
    from ragpipe.utils.costs import CostTracker

    tracker = CostTracker()
    tracker.record_generation("gpt-4o", prompt_tokens=500, completion_tokens=200)
    tracker.record_embedding("text-embedding-3-small", token_count=1000)

    print(tracker.total_cost)       # $0.0035
    print(tracker.summary())        # breakdown by model
    print(tracker.to_dict())        # JSON-serializable
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# Pricing per 1M tokens (April 2026 estimates, USD)
# Format: {model_prefix: (input_price_per_1M, output_price_per_1M)}
DEFAULT_GENERATION_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-5.4": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-4.6": (3.00, 15.00),
    "claude-4": (3.00, 15.00),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Google
    "gemini-3.1": (1.25, 5.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    # Local (free)
    "ollama": (0.0, 0.0),
    "gemma": (0.0, 0.0),
    "llama": (0.0, 0.0),
    "qwen": (0.0, 0.0),
    "deepseek": (0.0, 0.0),
    "phi": (0.0, 0.0),
    "mistral-small": (0.0, 0.0),
}

# Embedding pricing per 1M tokens
DEFAULT_EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
    "voyage-3": 0.06,
    "voyage-code-3": 0.18,
    "jina-embeddings-v3": 0.02,
    # Local (free)
    "ollama": 0.0,
    "nomic-embed": 0.0,
    "all-MiniLM": 0.0,
    "sentence-transformers": 0.0,
}


@dataclass
class UsageRecord:
    """A single usage record for cost tracking."""
    timestamp: float
    model: str
    operation: str  # "generation" or "embedding"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Track and aggregate LLM API costs across queries.

    Supports custom pricing, budget enforcement, and per-model breakdowns.
    """

    def __init__(
        self,
        generation_pricing: dict[str, tuple[float, float]] | None = None,
        embedding_pricing: dict[str, float] | None = None,
        budget_usd: float | None = None,
    ):
        self._generation_pricing = generation_pricing or DEFAULT_GENERATION_PRICING
        self._embedding_pricing = embedding_pricing or DEFAULT_EMBEDDING_PRICING
        self._budget_usd = budget_usd
        self._records: list[UsageRecord] = []

    def _find_price(self, model: str, pricing_dict: dict) -> Any:
        """Find pricing by matching model name prefix."""
        model_lower = model.lower()
        for prefix, price in pricing_dict.items():
            if model_lower.startswith(prefix.lower()) or prefix.lower() in model_lower:
                return price
        return None

    def record_generation(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        **metadata: Any,
    ) -> UsageRecord:
        """Record a generation API call and compute cost."""
        total = prompt_tokens + completion_tokens
        price = self._find_price(model, self._generation_pricing)

        cost = 0.0
        if price:
            input_price, output_price = price
            cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            operation="generation",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=cost,
            metadata=metadata,
        )
        self._records.append(record)
        return record

    def record_embedding(
        self,
        model: str,
        token_count: int = 0,
        **metadata: Any,
    ) -> UsageRecord:
        """Record an embedding API call and compute cost."""
        price = self._find_price(model, self._embedding_pricing)
        cost = (token_count * price / 1_000_000) if price else 0.0

        record = UsageRecord(
            timestamp=time.time(),
            model=model,
            operation="embedding",
            total_tokens=token_count,
            cost_usd=cost,
            metadata=metadata,
        )
        self._records.append(record)
        return record

    @property
    def total_cost(self) -> float:
        """Total cost in USD across all records."""
        return sum(r.cost_usd for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all records."""
        return sum(r.total_tokens for r in self._records)

    @property
    def query_count(self) -> int:
        """Number of recorded operations."""
        return len(self._records)

    @property
    def is_over_budget(self) -> bool:
        """Check if total cost exceeds budget."""
        if self._budget_usd is None:
            return False
        return self.total_cost > self._budget_usd

    @property
    def remaining_budget(self) -> float | None:
        """Remaining budget in USD, or None if no budget set."""
        if self._budget_usd is None:
            return None
        return max(0.0, self._budget_usd - self.total_cost)

    def cost_by_model(self) -> dict[str, float]:
        """Aggregate cost by model."""
        by_model: dict[str, float] = {}
        for r in self._records:
            by_model[r.model] = by_model.get(r.model, 0.0) + r.cost_usd
        return by_model

    def cost_by_operation(self) -> dict[str, float]:
        """Aggregate cost by operation type."""
        by_op: dict[str, float] = {}
        for r in self._records:
            by_op[r.operation] = by_op.get(r.operation, 0.0) + r.cost_usd
        return by_op

    def tokens_by_model(self) -> dict[str, int]:
        """Aggregate tokens by model."""
        by_model: dict[str, int] = {}
        for r in self._records:
            by_model[r.model] = by_model.get(r.model, 0) + r.total_tokens
        return by_model

    def summary(self) -> str:
        """Human-readable cost summary."""
        lines = [
            f"Cost Summary ({self.query_count} operations, {self.total_tokens:,} tokens, ${self.total_cost:.6f})",
        ]
        if self._budget_usd is not None:
            pct = (self.total_cost / self._budget_usd * 100) if self._budget_usd > 0 else 0
            lines.append(f"  Budget: ${self._budget_usd:.4f} | Used: {pct:.1f}% | Remaining: ${self.remaining_budget:.4f}")

        by_model = self.cost_by_model()
        if by_model:
            lines.append("  By model:")
            for model, cost in sorted(by_model.items(), key=lambda x: -x[1]):
                tokens = self.tokens_by_model().get(model, 0)
                lines.append(f"    {model}: ${cost:.6f} ({tokens:,} tokens)")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable cost report."""
        return {
            "total_cost_usd": round(self.total_cost, 8),
            "total_tokens": self.total_tokens,
            "query_count": self.query_count,
            "budget_usd": self._budget_usd,
            "remaining_budget_usd": self.remaining_budget,
            "is_over_budget": self.is_over_budget,
            "cost_by_model": {k: round(v, 8) for k, v in self.cost_by_model().items()},
            "cost_by_operation": {k: round(v, 8) for k, v in self.cost_by_operation().items()},
            "records": [
                {
                    "model": r.model,
                    "operation": r.operation,
                    "total_tokens": r.total_tokens,
                    "cost_usd": round(r.cost_usd, 8),
                }
                for r in self._records
            ],
        }

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()
