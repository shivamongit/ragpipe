"""Context Window — programmable context composition for RAG.

Treats the LLM context window as a structured, composable interface.
Instead of "retrieve top-K and stuff into prompt", provides operators for
prioritization, compression, deduplication, conflict resolution,
freshness weighting, and token budgeting.

Usage:
    from ragpipe.context import ContextWindow

    ctx = ContextWindow(max_tokens=4096)
    ctx.add_retrieval_results(results)
    ctx.deduplicate(similarity_threshold=0.9)
    ctx.prioritize("relevance")
    ctx.budget(max_tokens=3000)
    prompt_text = ctx.render()
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ContextStrategy(str, Enum):
    """Strategies for context prioritization."""
    RELEVANCE = "relevance"       # Sort by retrieval score (default)
    DIVERSITY = "diversity"       # Maximize topic diversity
    RECENCY = "recency"           # Prefer newer documents
    POSITION = "position"         # Original document order
    DENSITY = "density"           # Prefer information-dense chunks


@dataclass
class ContextItem:
    """A single item in the context window with rich metadata."""
    text: str
    score: float = 0.0
    source: str = ""
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    is_duplicate: bool = False
    conflict_group: str = ""
    priority: float = 0.0

    def __post_init__(self):
        if not self.token_count:
            self.token_count = self._estimate_tokens(self.text)
        if not self.priority:
            self.priority = self.score

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return max(1, len(text) // 4)


class ContextWindow:
    """Programmable context composition engine.

    Provides structured operators for building optimal LLM context windows
    from retrieval results. This is the core primitive for context engineering.

    Key operations:
        - add/remove items
        - deduplicate (exact + near-duplicate)
        - prioritize (by score, recency, diversity, density)
        - compress (summarize long chunks)
        - resolve_conflicts (handle contradicting sources)
        - budget (enforce token limits)
        - render (produce final prompt text)
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        separator: str = "\n\n---\n\n",
        source_format: str = "[Source {i}] (score: {score:.3f})\n{text}",
    ):
        self.max_tokens = max_tokens
        self.separator = separator
        self.source_format = source_format
        self._items: list[ContextItem] = []
        self._operations_log: list[str] = []

    # ── Add items ─────────────────────────────────────────────────────────────

    def add(self, text: str, score: float = 0.0, **kwargs) -> ContextWindow:
        """Add a single context item."""
        self._items.append(ContextItem(text=text, score=score, **kwargs))
        return self

    def add_retrieval_results(self, results: list) -> ContextWindow:
        """Add items from RetrievalResult objects."""
        for r in results:
            self._items.append(ContextItem(
                text=r.chunk.text,
                score=r.score,
                chunk_id=r.chunk.id,
                source=r.chunk.metadata.get("source", ""),
                metadata=r.chunk.metadata,
            ))
        self._operations_log.append(f"added {len(results)} retrieval results")
        return self

    def add_items(self, items: list[ContextItem]) -> ContextWindow:
        """Add pre-built ContextItem objects."""
        self._items.extend(items)
        return self

    # ── Deduplication ─────────────────────────────────────────────────────────

    def deduplicate(
        self,
        similarity_threshold: float = 0.85,
        method: str = "jaccard",
    ) -> ContextWindow:
        """Remove duplicate and near-duplicate items.

        Methods:
            - "exact": Remove items with identical text
            - "jaccard": Remove items with Jaccard word similarity > threshold
            - "hash": Remove items with identical content hash
        """
        if method == "exact":
            seen: set[str] = set()
            unique = []
            for item in self._items:
                if item.text not in seen:
                    seen.add(item.text)
                    unique.append(item)
                else:
                    item.is_duplicate = True
            removed = len(self._items) - len(unique)
            self._items = unique

        elif method == "hash":
            seen_hashes: set[str] = set()
            unique = []
            for item in self._items:
                h = hashlib.md5(item.text.strip().lower().encode()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique.append(item)
            removed = len(self._items) - len(unique)
            self._items = unique

        else:  # jaccard
            unique = []
            for item in self._items:
                is_dup = False
                item_words = set(item.text.lower().split())
                for existing in unique:
                    existing_words = set(existing.text.lower().split())
                    intersection = item_words & existing_words
                    union = item_words | existing_words
                    if union and len(intersection) / len(union) > similarity_threshold:
                        is_dup = True
                        break
                if not is_dup:
                    unique.append(item)
            removed = len(self._items) - len(unique)
            self._items = unique

        self._operations_log.append(f"deduplicate({method}): removed {removed}")
        return self

    # ── Prioritization ────────────────────────────────────────────────────────

    def prioritize(self, strategy: str | ContextStrategy = "relevance") -> ContextWindow:
        """Sort items by the given prioritization strategy."""
        if isinstance(strategy, str):
            strategy = ContextStrategy(strategy)

        if strategy == ContextStrategy.RELEVANCE:
            self._items.sort(key=lambda x: x.score, reverse=True)

        elif strategy == ContextStrategy.RECENCY:
            def _recency_key(item: ContextItem) -> float:
                ts = item.metadata.get("timestamp", item.metadata.get("date", 0))
                if isinstance(ts, str):
                    return hash(ts)
                return float(ts)
            self._items.sort(key=_recency_key, reverse=True)

        elif strategy == ContextStrategy.DIVERSITY:
            self._diversify()

        elif strategy == ContextStrategy.DENSITY:
            # Prefer chunks with more unique information per token
            def _density(item: ContextItem) -> float:
                words = set(item.text.lower().split())
                return len(words) / max(1, item.token_count)
            self._items.sort(key=_density, reverse=True)

        elif strategy == ContextStrategy.POSITION:
            def _position_key(item: ContextItem) -> str:
                return item.chunk_id or ""
            self._items.sort(key=_position_key)

        self._operations_log.append(f"prioritize({strategy.value})")
        return self

    def _diversify(self) -> None:
        """MMR-style diversification: greedily select items maximizing diversity."""
        if len(self._items) <= 1:
            return

        selected = [self._items[0]]  # Start with highest-scored
        remaining = list(self._items[1:])

        while remaining:
            best_idx = 0
            best_score = -1.0
            for i, candidate in enumerate(remaining):
                # Similarity to already selected (lower is better for diversity)
                max_sim = 0.0
                cand_words = set(candidate.text.lower().split())
                for sel in selected:
                    sel_words = set(sel.text.lower().split())
                    union = cand_words | sel_words
                    if union:
                        sim = len(cand_words & sel_words) / len(union)
                        max_sim = max(max_sim, sim)
                # MMR: λ * relevance - (1-λ) * max_similarity
                mmr = 0.7 * candidate.score - 0.3 * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        self._items = selected

    # ── Compression ───────────────────────────────────────────────────────────

    def compress(
        self,
        compress_fn: Callable[[str], str] | None = None,
        max_item_tokens: int | None = None,
    ) -> ContextWindow:
        """Compress context items to fit within token budget.

        Args:
            compress_fn: Optional LLM function to summarize text.
            max_item_tokens: Truncate items longer than this.
        """
        for item in self._items:
            if max_item_tokens and item.token_count > max_item_tokens:
                if compress_fn:
                    item.text = compress_fn(item.text)
                    item.token_count = ContextItem._estimate_tokens(item.text)
                else:
                    # Simple truncation
                    char_limit = max_item_tokens * 4
                    if len(item.text) > char_limit:
                        item.text = item.text[:char_limit] + "..."
                        item.token_count = max_item_tokens

        self._operations_log.append(f"compress(max_item_tokens={max_item_tokens})")
        return self

    # ── Conflict Resolution ───────────────────────────────────────────────────

    def resolve_conflicts(
        self,
        resolve_fn: Callable[[list[ContextItem]], ContextItem] | None = None,
    ) -> ContextWindow:
        """Detect and resolve conflicting information across items.

        If resolve_fn is not provided, keeps the higher-scored item from
        each conflict group. Conflict groups are auto-detected based on
        overlapping topics in the same source.
        """
        if not resolve_fn:
            # Default: keep highest-scored item per source
            seen_sources: dict[str, ContextItem] = {}
            resolved = []
            for item in self._items:
                if item.source and item.source in seen_sources:
                    existing = seen_sources[item.source]
                    if item.score > existing.score:
                        resolved = [i for i in resolved if i is not existing]
                        resolved.append(item)
                        seen_sources[item.source] = item
                else:
                    resolved.append(item)
                    if item.source:
                        seen_sources[item.source] = item
            self._items = resolved
        else:
            # Group by source, resolve each group
            groups: dict[str, list[ContextItem]] = {}
            ungrouped = []
            for item in self._items:
                key = item.source or item.chunk_id
                if key:
                    groups.setdefault(key, []).append(item)
                else:
                    ungrouped.append(item)
            resolved = ungrouped[:]
            for group_items in groups.values():
                if len(group_items) > 1:
                    resolved.append(resolve_fn(group_items))
                else:
                    resolved.extend(group_items)
            self._items = resolved

        self._operations_log.append("resolve_conflicts")
        return self

    # ── Token Budgeting ───────────────────────────────────────────────────────

    def budget(self, max_tokens: int | None = None) -> ContextWindow:
        """Enforce a token budget by removing lowest-priority items.

        Items are removed from the end (lowest priority after sorting)
        until the total fits within the budget.
        """
        limit = max_tokens or self.max_tokens
        total = sum(item.token_count for item in self._items)

        if total <= limit:
            self._operations_log.append(f"budget({limit}): fits ({total} tokens)")
            return self

        # Remove items from the end until within budget
        kept = []
        running = 0
        for item in self._items:
            if running + item.token_count <= limit:
                kept.append(item)
                running += item.token_count
            else:
                break

        removed = len(self._items) - len(kept)
        self._items = kept
        self._operations_log.append(f"budget({limit}): removed {removed}, kept {len(kept)}")
        return self

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, format: str | None = None) -> str:
        """Render the context window as a string for LLM prompting.

        Args:
            format: Optional format string. Use {i}, {text}, {score}, {source}.
        """
        fmt = format or self.source_format
        parts = []
        for i, item in enumerate(self._items):
            rendered = fmt.format(
                i=i + 1,
                text=item.text,
                score=item.score,
                source=item.source or f"chunk-{i+1}",
                chunk_id=item.chunk_id,
            )
            parts.append(rendered)
        return self.separator.join(parts)

    def render_citations(self) -> list[dict[str, Any]]:
        """Return structured citations for each context item."""
        return [
            {
                "index": i + 1,
                "text": item.text[:200],
                "score": item.score,
                "source": item.source,
                "chunk_id": item.chunk_id,
                "token_count": item.token_count,
            }
            for i, item in enumerate(self._items)
        ]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def items(self) -> list[ContextItem]:
        """Current context items."""
        return list(self._items)

    @property
    def total_tokens(self) -> int:
        """Total estimated tokens in the context window."""
        return sum(item.token_count for item in self._items)

    @property
    def item_count(self) -> int:
        """Number of items in the context window."""
        return len(self._items)

    @property
    def utilization(self) -> float:
        """Token utilization as a fraction of max_tokens."""
        if self.max_tokens <= 0:
            return 0.0
        return min(1.0, self.total_tokens / self.max_tokens)

    @property
    def operations(self) -> list[str]:
        """Log of operations applied to this context window."""
        return list(self._operations_log)

    # ── Filtering ─────────────────────────────────────────────────────────────

    def filter(self, predicate: Callable[[ContextItem], bool]) -> ContextWindow:
        """Keep only items matching the predicate."""
        before = len(self._items)
        self._items = [item for item in self._items if predicate(item)]
        self._operations_log.append(f"filter: {before} → {len(self._items)}")
        return self

    def filter_by_score(self, min_score: float = 0.0) -> ContextWindow:
        """Remove items below a minimum relevance score."""
        return self.filter(lambda item: item.score >= min_score)

    def filter_by_source(self, sources: list[str]) -> ContextWindow:
        """Keep only items from specific sources."""
        source_set = set(sources)
        return self.filter(lambda item: item.source in source_set)

    # ── Chaining ──────────────────────────────────────────────────────────────

    def pipe(self, *operations: Callable) -> ContextWindow:
        """Apply a sequence of operations to the context window.

        Usage:
            ctx.pipe(
                lambda c: c.deduplicate(),
                lambda c: c.prioritize("relevance"),
                lambda c: c.budget(3000),
            )
        """
        result = self
        for op in operations:
            result = op(result)
        return result

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "item_count": self.item_count,
            "total_tokens": self.total_tokens,
            "max_tokens": self.max_tokens,
            "utilization": round(self.utilization, 3),
            "operations": self._operations_log,
            "items": [
                {
                    "text": item.text[:200],
                    "score": item.score,
                    "source": item.source,
                    "token_count": item.token_count,
                    "chunk_id": item.chunk_id,
                }
                for item in self._items
            ],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ContextWindow: {self.item_count} items, {self.total_tokens} tokens "
            f"({self.utilization:.0%} of {self.max_tokens} budget)",
        ]
        if self._operations_log:
            lines.append(f"  Operations: {' → '.join(self._operations_log)}")
        for i, item in enumerate(self._items[:5]):
            preview = item.text[:60].replace("\n", " ")
            lines.append(f"  [{i+1}] score={item.score:.3f} tokens={item.token_count} | {preview}...")
        if len(self._items) > 5:
            lines.append(f"  ... and {len(self._items) - 5} more")
        return "\n".join(lines)

    def clear(self) -> ContextWindow:
        """Remove all items and reset operations log."""
        self._items.clear()
        self._operations_log.clear()
        return self
