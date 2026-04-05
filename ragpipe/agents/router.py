"""Agentic RAG query router — classifies queries and orchestrates multi-step retrieval.

The router decides the optimal retrieval strategy for each query:
- DIRECT: answer from LLM knowledge (no retrieval needed)
- SINGLE: standard single-pass RAG retrieval
- MULTI_STEP: decompose complex queries into sub-questions, retrieve for each
- SUMMARIZE: retrieve broadly and synthesize across many chunks

This is the core of "Agentic RAG" — the LLM decides HOW to answer, not just WHAT.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ragpipe.core import GenerationResult, Pipeline, RetrievalResult


class RouteType(str, Enum):
    """Query routing strategies."""
    DIRECT = "direct"
    SINGLE = "single"
    MULTI_STEP = "multi_step"
    SUMMARIZE = "summarize"


@dataclass
class RouteDecision:
    """The router's decision on how to handle a query."""
    route: RouteType
    sub_questions: list[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Prompt for classification
ROUTER_PROMPT = """You are a query routing agent for a RAG system. Classify the user's question into one of these strategies:

1. DIRECT — The question can be answered from general knowledge without retrieving documents (e.g., "What is 2+2?", "What does RAG stand for?")
2. SINGLE — Standard retrieval: embed the question, find relevant chunks, generate answer (most questions)
3. MULTI_STEP — Complex question that should be decomposed into 2-4 sub-questions, each retrieved separately, then synthesized (e.g., "Compare X and Y in terms of A, B, and C")
4. SUMMARIZE — Question requires broad coverage across many documents (e.g., "Give me an overview of all findings", "Summarize the key themes")

Respond with ONLY a JSON object:
{
  "route": "direct" | "single" | "multi_step" | "summarize",
  "sub_questions": ["q1", "q2", ...],  // only for multi_step, otherwise []
  "reasoning": "brief explanation"
}

Question: """


def _parse_route_response(text: str) -> RouteDecision:
    """Parse the LLM's JSON response into a RouteDecision."""
    text = text.strip()

    # Try to extract JSON from the response
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            route_str = data.get("route", "single").lower().strip()

            route_map = {
                "direct": RouteType.DIRECT,
                "single": RouteType.SINGLE,
                "multi_step": RouteType.MULTI_STEP,
                "summarize": RouteType.SUMMARIZE,
            }
            route = route_map.get(route_str, RouteType.SINGLE)

            return RouteDecision(
                route=route,
                sub_questions=data.get("sub_questions", []),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: default to single retrieval
    return RouteDecision(route=RouteType.SINGLE, reasoning="Failed to parse route; defaulting to single retrieval")


class QueryRouter:
    """Agentic RAG router that intelligently routes queries to optimal strategies.

    The router uses an LLM to classify queries and then orchestrates the
    appropriate retrieval strategy via the Pipeline.

    Usage:
        router = QueryRouter(pipeline=pipe, classify_fn=my_llm_call)
        result = await router.aquery("Compare FAISS and ChromaDB performance")

    The classify_fn should accept a prompt string and return a string response.
    If no classify_fn is provided, all queries default to SINGLE retrieval.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        classify_fn: Optional[Callable[[str], str]] = None,
        aclassify_fn: Optional[Callable] = None,
        max_sub_questions: int = 4,
        summarize_top_k: int = 15,
    ):
        self.pipeline = pipeline
        self._classify_fn = classify_fn
        self._aclassify_fn = aclassify_fn
        self.max_sub_questions = max_sub_questions
        self.summarize_top_k = summarize_top_k

    def classify(self, question: str) -> RouteDecision:
        """Classify a query into a routing strategy (sync)."""
        if not self._classify_fn:
            return RouteDecision(route=RouteType.SINGLE, reasoning="No classify_fn provided")

        prompt = ROUTER_PROMPT + question
        response = self._classify_fn(prompt)
        decision = _parse_route_response(response)

        # Enforce max sub-questions
        if decision.sub_questions:
            decision.sub_questions = decision.sub_questions[:self.max_sub_questions]

        return decision

    async def aclassify(self, question: str) -> RouteDecision:
        """Classify a query into a routing strategy (async)."""
        if self._aclassify_fn:
            prompt = ROUTER_PROMPT + question
            response = await self._aclassify_fn(prompt)
            decision = _parse_route_response(response)
        elif self._classify_fn:
            decision = await asyncio.to_thread(self.classify, question)
        else:
            return RouteDecision(route=RouteType.SINGLE, reasoning="No classify_fn provided")

        if decision.sub_questions:
            decision.sub_questions = decision.sub_questions[:self.max_sub_questions]

        return decision

    def query(self, question: str, top_k: int | None = None) -> GenerationResult:
        """Route and execute a query (sync)."""
        from ragpipe.core import GenerationResult

        decision = self.classify(question)

        if decision.route == RouteType.DIRECT:
            # Let the generator answer without context
            answer = self.pipeline.generator.generate(question, [])
            return GenerationResult(
                answer=answer.answer,
                sources=[],
                model=answer.model,
                tokens_used=answer.tokens_used,
                metadata={"route": "direct", "reasoning": decision.reasoning},
            )

        elif decision.route == RouteType.MULTI_STEP:
            return self._multi_step_query(question, decision, top_k)

        elif decision.route == RouteType.SUMMARIZE:
            return self.pipeline.query(question, top_k=self.summarize_top_k)

        else:  # SINGLE
            result = self.pipeline.query(question, top_k=top_k)
            result.metadata["route"] = "single"
            return result

    async def aquery(self, question: str, top_k: int | None = None) -> GenerationResult:
        """Route and execute a query (async)."""
        from ragpipe.core import GenerationResult

        decision = await self.aclassify(question)

        if decision.route == RouteType.DIRECT:
            answer = await self.pipeline.generator.agenerate(question, [])
            return GenerationResult(
                answer=answer.answer,
                sources=[],
                model=answer.model,
                tokens_used=answer.tokens_used,
                metadata={"route": "direct", "reasoning": decision.reasoning},
            )

        elif decision.route == RouteType.MULTI_STEP:
            return await self._amulti_step_query(question, decision, top_k)

        elif decision.route == RouteType.SUMMARIZE:
            result = await self.pipeline.aquery(question, top_k=self.summarize_top_k)
            result.metadata["route"] = "summarize"
            return result

        else:  # SINGLE
            result = await self.pipeline.aquery(question, top_k=top_k)
            result.metadata["route"] = "single"
            return result

    def _multi_step_query(
        self, original_question: str, decision: RouteDecision, top_k: int | None
    ) -> GenerationResult:
        """Execute multi-step retrieval: retrieve for each sub-question, merge, generate."""
        import time
        from ragpipe.core import GenerationResult

        t0 = time.perf_counter()

        # Retrieve for each sub-question
        all_results: list = []
        seen_chunk_ids: set[str] = set()

        for sub_q in decision.sub_questions:
            results = self.pipeline.retrieve(sub_q, top_k=top_k)
            for r in results:
                if r.chunk.id not in seen_chunk_ids:
                    seen_chunk_ids.add(r.chunk.id)
                    all_results.append(r)

        # Sort by score and take top results
        all_results.sort(key=lambda r: r.score, reverse=True)
        top_results = all_results[:self.pipeline.top_k * 2]

        # Rerank merged results if reranker is available
        if self.pipeline.reranker and top_results:
            top_results = self.pipeline.reranker.rerank(
                original_question, top_results, top_k=self.pipeline.rerank_top_k
            )

        for i, r in enumerate(top_results):
            r.rank = i + 1

        # Generate with merged context
        answer = self.pipeline.generator.generate(original_question, top_results)

        latency = (time.perf_counter() - t0) * 1000
        return GenerationResult(
            answer=answer.answer,
            sources=top_results,
            model=answer.model,
            tokens_used=answer.tokens_used,
            latency_ms=round(latency, 2),
            metadata={
                "route": "multi_step",
                "sub_questions": decision.sub_questions,
                "total_unique_chunks": len(all_results),
                "reasoning": decision.reasoning,
            },
        )

    async def _amulti_step_query(
        self, original_question: str, decision: RouteDecision, top_k: int | None
    ) -> GenerationResult:
        """Async multi-step retrieval with parallel sub-question retrieval."""
        import time
        from ragpipe.core import GenerationResult

        t0 = time.perf_counter()

        # Retrieve for all sub-questions in parallel
        tasks = [self.pipeline.aretrieve(sub_q, top_k=top_k) for sub_q in decision.sub_questions]
        sub_results = await asyncio.gather(*tasks)

        # Merge and deduplicate
        all_results: list = []
        seen_chunk_ids: set[str] = set()

        for results in sub_results:
            for r in results:
                if r.chunk.id not in seen_chunk_ids:
                    seen_chunk_ids.add(r.chunk.id)
                    all_results.append(r)

        all_results.sort(key=lambda r: r.score, reverse=True)
        top_results = all_results[:self.pipeline.top_k * 2]

        if self.pipeline.reranker and top_results:
            top_results = await self.pipeline.reranker.arerank(
                original_question, top_results, top_k=self.pipeline.rerank_top_k
            )

        for i, r in enumerate(top_results):
            r.rank = i + 1

        answer = await self.pipeline.generator.agenerate(original_question, top_results)

        latency = (time.perf_counter() - t0) * 1000
        return GenerationResult(
            answer=answer.answer,
            sources=top_results,
            model=answer.model,
            tokens_used=answer.tokens_used,
            latency_ms=round(latency, 2),
            metadata={
                "route": "multi_step",
                "sub_questions": decision.sub_questions,
                "total_unique_chunks": len(all_results),
                "reasoning": decision.reasoning,
            },
        )
