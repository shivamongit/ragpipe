"""Adaptive Retrieval — dynamically switches retrieval strategy based on query analysis.

No other framework provides built-in adaptive retrieval that automatically:
1. Classifies query complexity (factual / analytical / comparative / exploratory)
2. Selects the optimal retrieval strategy (dense, sparse, hybrid, multi-pass)
3. Adjusts top_k dynamically based on query type
4. Scores retrieval confidence and retries with a different strategy if low

Usage:
    from ragpipe.agents import AdaptiveRetriever

    retriever = AdaptiveRetriever(
        classify_fn=my_llm,
        strategies={"dense": dense_fn, "sparse": sparse_fn, "hybrid": hybrid_fn},
        rerank_fn=my_reranker,
    )
    result = retriever.retrieve("Compare FAISS vs ChromaDB performance")
    print(result.documents, result.strategy_used, result.confidence)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class QueryComplexity(str, Enum):
    """Classified query complexity levels."""
    FACTUAL = "factual"           # Simple fact lookup → dense, low top_k
    ANALYTICAL = "analytical"     # Needs reasoning → hybrid, medium top_k
    COMPARATIVE = "comparative"   # Compare multiple things → multi-pass, high top_k
    EXPLORATORY = "exploratory"   # Broad overview → sparse + dense, high top_k
    CONVERSATIONAL = "conversational"  # Follow-up → dense with context


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    MULTI_PASS = "multi_pass"


@dataclass
class AdaptiveResult:
    """Result from adaptive retrieval."""
    documents: list[Any]
    strategy_used: RetrievalStrategy
    query_complexity: QueryComplexity
    confidence: float
    top_k_used: int
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# Strategy selection matrix
STRATEGY_MAP: dict[QueryComplexity, tuple[RetrievalStrategy, int]] = {
    QueryComplexity.FACTUAL: (RetrievalStrategy.DENSE, 3),
    QueryComplexity.ANALYTICAL: (RetrievalStrategy.HYBRID, 5),
    QueryComplexity.COMPARATIVE: (RetrievalStrategy.MULTI_PASS, 8),
    QueryComplexity.EXPLORATORY: (RetrievalStrategy.HYBRID, 10),
    QueryComplexity.CONVERSATIONAL: (RetrievalStrategy.DENSE, 5),
}

# Fallback chain: if one strategy fails, try the next
FALLBACK_CHAIN: dict[RetrievalStrategy, RetrievalStrategy] = {
    RetrievalStrategy.DENSE: RetrievalStrategy.HYBRID,
    RetrievalStrategy.SPARSE: RetrievalStrategy.HYBRID,
    RetrievalStrategy.HYBRID: RetrievalStrategy.MULTI_PASS,
    RetrievalStrategy.MULTI_PASS: RetrievalStrategy.HYBRID,
}

CLASSIFY_PROMPT = """Classify this search query into one complexity category:

- FACTUAL: Simple fact lookup ("What is X?", "When did Y happen?", "Who is Z?")
- ANALYTICAL: Needs reasoning or explanation ("Why does X happen?", "How does Y work?")
- COMPARATIVE: Compares multiple items ("Compare X vs Y", "Differences between A and B")
- EXPLORATORY: Broad overview or survey ("Overview of X", "Key trends in Y", "Summarize Z")
- CONVERSATIONAL: Follow-up or contextual ("What about the second point?", "Tell me more")

Query: {query}

Respond with ONLY a JSON object:
{{"complexity": "factual|analytical|comparative|exploratory|conversational", "reasoning": "brief explanation"}}"""


def _parse_complexity(raw: str) -> QueryComplexity:
    """Parse LLM classification into QueryComplexity."""
    raw = raw.strip().lower()
    try:
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            c = data.get("complexity", "").lower().strip()
            for member in QueryComplexity:
                if member.value == c:
                    return member
    except (json.JSONDecodeError, ValueError):
        pass

    # Keyword fallback
    if "compar" in raw:
        return QueryComplexity.COMPARATIVE
    elif "explor" in raw or "overview" in raw or "summar" in raw:
        return QueryComplexity.EXPLORATORY
    elif "analy" in raw or "why" in raw or "how" in raw:
        return QueryComplexity.ANALYTICAL
    elif "conversat" in raw or "follow" in raw:
        return QueryComplexity.CONVERSATIONAL
    return QueryComplexity.FACTUAL


class AdaptiveRetriever:
    """Retriever that automatically adapts strategy based on query analysis.

    This is a unique differentiator — no other RAG framework provides built-in
    adaptive retrieval that classifies queries, selects strategies, adjusts top_k,
    and retries with fallback strategies on low confidence.
    """

    def __init__(
        self,
        classify_fn: Optional[Callable] = None,
        strategies: Optional[dict[str, Callable]] = None,
        rerank_fn: Optional[Callable] = None,
        confidence_threshold: float = 0.3,
        max_retries: int = 1,
    ):
        self.classify_fn = classify_fn
        self.strategies = strategies or {}
        self.rerank_fn = rerank_fn
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries

    def _classify_query(self, query: str) -> QueryComplexity:
        """Classify query complexity using LLM or heuristics."""
        if self.classify_fn is None:
            return self._heuristic_classify(query)
        prompt = CLASSIFY_PROMPT.format(query=query)
        raw = self.classify_fn(prompt)
        return _parse_complexity(raw)

    @staticmethod
    def _heuristic_classify(query: str) -> QueryComplexity:
        """Fast rule-based classification when no LLM is available."""
        q = query.lower().strip()
        # Comparative patterns
        if any(w in q for w in ["compare", "versus", " vs ", "difference between", "better than"]):
            return QueryComplexity.COMPARATIVE
        # Exploratory patterns
        if any(w in q for w in ["overview", "summarize", "key themes", "list all", "what are the"]):
            return QueryComplexity.EXPLORATORY
        # Analytical patterns
        if any(w in q for w in ["why ", "how does", "explain", "what causes", "reason for"]):
            return QueryComplexity.ANALYTICAL
        # Conversational patterns
        if any(w in q for w in ["what about", "tell me more", "and the", "also "]):
            return QueryComplexity.CONVERSATIONAL
        # Default to factual
        return QueryComplexity.FACTUAL

    def _select_strategy(
        self, complexity: QueryComplexity,
    ) -> tuple[RetrievalStrategy, int]:
        """Select retrieval strategy and top_k based on query complexity."""
        return STRATEGY_MAP.get(
            complexity, (RetrievalStrategy.HYBRID, 5),
        )

    def _execute_strategy(
        self, query: str, strategy: RetrievalStrategy, top_k: int,
    ) -> list[Any]:
        """Execute a retrieval strategy."""
        strategy_name = strategy.value
        if strategy == RetrievalStrategy.MULTI_PASS:
            # Multi-pass: try dense then sparse, merge results
            results = []
            if "dense" in self.strategies:
                results.extend(self.strategies["dense"](query, top_k=top_k))
            if "sparse" in self.strategies:
                results.extend(self.strategies["sparse"](query, top_k=top_k))
            # Deduplicate by string representation
            seen = set()
            unique = []
            for r in results:
                key = str(r)[:200]
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            return unique[:top_k]

        if strategy_name in self.strategies:
            return self.strategies[strategy_name](query, top_k=top_k)

        # Fallback: try any available strategy
        for name, fn in self.strategies.items():
            return fn(query, top_k=top_k)
        return []

    def _score_confidence(self, documents: list[Any]) -> float:
        """Score retrieval confidence based on result quality signals."""
        if not documents:
            return 0.0

        # Base confidence from having results
        confidence = min(len(documents) / 3, 1.0) * 0.5

        # If documents have scores, use them
        scores = []
        for doc in documents:
            if hasattr(doc, 'score'):
                scores.append(doc.score)
            elif isinstance(doc, dict) and 'score' in doc:
                scores.append(doc['score'])

        if scores:
            avg_score = sum(scores) / len(scores)
            confidence += avg_score * 0.5

        return min(confidence, 1.0)

    def retrieve(self, query: str, **kwargs: Any) -> AdaptiveResult:
        """Adaptively retrieve documents with automatic strategy selection."""
        if not self.strategies:
            return AdaptiveResult(
                documents=[],
                strategy_used=RetrievalStrategy.DENSE,
                query_complexity=QueryComplexity.FACTUAL,
                confidence=0.0,
                top_k_used=0,
            )

        # Step 1: Classify query
        complexity = self._classify_query(query)

        # Step 2: Select strategy
        strategy, top_k = self._select_strategy(complexity)
        top_k = kwargs.get("top_k", top_k)

        # Step 3: Execute with retry loop
        retries = 0
        current_strategy = strategy
        documents = self._execute_strategy(query, current_strategy, top_k)
        confidence = self._score_confidence(documents)

        while confidence < self.confidence_threshold and retries < self.max_retries:
            retries += 1
            fallback = FALLBACK_CHAIN.get(current_strategy)
            if fallback and fallback != current_strategy:
                current_strategy = fallback
                documents = self._execute_strategy(query, current_strategy, top_k + 2)
                confidence = self._score_confidence(documents)

        # Step 4: Optional reranking
        if self.rerank_fn and documents:
            documents = self.rerank_fn(query, documents)

        return AdaptiveResult(
            documents=documents,
            strategy_used=current_strategy,
            query_complexity=complexity,
            confidence=confidence,
            top_k_used=top_k,
            retries=retries,
        )

    async def aretrieve(self, query: str, **kwargs: Any) -> AdaptiveResult:
        """Async version of retrieve."""
        return await asyncio.to_thread(self.retrieve, query, **kwargs)
