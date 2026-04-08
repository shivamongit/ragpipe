"""Agentic Retrieval — multi-agent system for complex query handling.

Decomposes complex queries into retrieval plans, executes them with
appropriate strategies, evaluates results, and synthesizes final answers.

Components:
    - RetrievalPlanner: Decomposes queries into multi-step retrieval plans
    - RetrievalEvaluator: Scores retrieval quality and decides if more is needed
    - RetrievalCritic: Reviews and critiques the synthesized answer
    - AgenticPipeline: Orchestrates the full plan→retrieve→evaluate→critique→synthesize loop

Usage:
    from ragpipe.agents.planner import AgenticPipeline

    agent = AgenticPipeline(
        retrieve_fn=pipeline.retrieve,
        generate_fn=lambda q, ctx: generator.generate(q, ctx).answer,
    )
    result = agent.run("Compare financial performance of A and B over 3 quarters")
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class PlanStepType(str, Enum):
    """Types of retrieval plan steps."""
    SEARCH = "search"
    FILTER = "filter"
    COMPARE = "compare"
    AGGREGATE = "aggregate"
    VERIFY = "verify"


@dataclass
class PlanStep:
    """A single step in a retrieval plan."""
    step_id: int
    query: str
    step_type: PlanStepType = PlanStepType.SEARCH
    depends_on: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "query": self.query,
            "step_type": self.step_type.value,
            "depends_on": self.depends_on,
            "score": round(self.score, 4),
        }


@dataclass
class RetrievalPlan:
    """A multi-step retrieval plan for complex queries."""
    original_query: str
    steps: list[PlanStep] = field(default_factory=list)
    reasoning: str = ""
    estimated_hops: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "steps": [s.to_dict() for s in self.steps],
            "reasoning": self.reasoning,
            "estimated_hops": self.estimated_hops,
        }


@dataclass
class AgenticResult:
    """Result from the agentic retrieval pipeline."""
    answer: str
    plan: RetrievalPlan | None = None
    retrieval_rounds: int = 0
    total_chunks_retrieved: int = 0
    critique: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "plan": self.plan.to_dict() if self.plan else None,
            "retrieval_rounds": self.retrieval_rounds,
            "total_chunks_retrieved": self.total_chunks_retrieved,
            "critique": self.critique,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
        }

    def summary(self) -> str:
        lines = [
            f"Agentic Result (confidence={self.confidence:.2f}, "
            f"rounds={self.retrieval_rounds}, chunks={self.total_chunks_retrieved})",
            f"  Answer: {self.answer[:200]}...",
        ]
        if self.plan:
            lines.append(f"  Plan: {len(self.plan.steps)} steps")
            for step in self.plan.steps:
                lines.append(f"    [{step.step_id}] {step.step_type.value}: {step.query[:60]}")
        if self.critique:
            lines.append(f"  Critique: {self.critique[:100]}")
        return "\n".join(lines)


class RetrievalPlanner:
    """Decomposes complex queries into multi-step retrieval plans.

    Can use an LLM for planning or fall back to heuristic decomposition.
    """

    def __init__(self, plan_fn: Callable[[str], str] | None = None):
        self._plan_fn = plan_fn

    def plan(self, query: str) -> RetrievalPlan:
        """Create a retrieval plan for the given query."""
        if self._plan_fn:
            return self._llm_plan(query)
        return self._heuristic_plan(query)

    def _heuristic_plan(self, query: str) -> RetrievalPlan:
        """Heuristic decomposition based on query structure."""
        steps: list[PlanStep] = []
        query_lower = query.lower()

        # Detect comparison queries
        compare_words = ["compare", "versus", "vs", "difference between", "contrast"]
        if any(w in query_lower for w in compare_words):
            # Extract entities to compare
            parts = re.split(r'\band\b|\bvs\.?\b|\bversus\b', query, flags=re.IGNORECASE)
            for i, part in enumerate(parts):
                part = part.strip().strip('.,?!')
                if len(part) > 3:
                    steps.append(PlanStep(
                        step_id=i + 1,
                        query=f"Information about {part}",
                        step_type=PlanStepType.SEARCH,
                    ))
            if len(steps) >= 2:
                steps.append(PlanStep(
                    step_id=len(steps) + 1,
                    query=query,
                    step_type=PlanStepType.COMPARE,
                    depends_on=[s.step_id for s in steps],
                ))
                return RetrievalPlan(
                    original_query=query,
                    steps=steps,
                    reasoning="Comparison query: search each entity separately, then compare",
                    estimated_hops=2,
                )

        # Detect multi-hop queries
        multi_hop_words = ["then", "after that", "based on", "using the", "which leads to"]
        if any(w in query_lower for w in multi_hop_words):
            sub_queries = re.split(
                r'\bthen\b|\bafter that\b|\bbased on\b',
                query, flags=re.IGNORECASE,
            )
            for i, sq in enumerate(sub_queries):
                sq = sq.strip().strip('.,?!')
                if len(sq) > 5:
                    deps = [i] if i > 0 else []
                    steps.append(PlanStep(
                        step_id=i + 1,
                        query=sq,
                        step_type=PlanStepType.SEARCH,
                        depends_on=deps,
                    ))
            if steps:
                return RetrievalPlan(
                    original_query=query,
                    steps=steps,
                    reasoning="Multi-hop query: sequential retrieval with dependencies",
                    estimated_hops=len(steps),
                )

        # Detect aggregation queries
        agg_words = ["how many", "total", "sum", "count", "all", "list all", "enumerate"]
        if any(w in query_lower for w in agg_words):
            steps.append(PlanStep(
                step_id=1, query=query, step_type=PlanStepType.SEARCH,
            ))
            steps.append(PlanStep(
                step_id=2, query=f"Aggregate results for: {query}",
                step_type=PlanStepType.AGGREGATE, depends_on=[1],
            ))
            return RetrievalPlan(
                original_query=query, steps=steps,
                reasoning="Aggregation query: search then aggregate",
                estimated_hops=2,
            )

        # Default: single-step search
        steps.append(PlanStep(step_id=1, query=query, step_type=PlanStepType.SEARCH))
        return RetrievalPlan(
            original_query=query, steps=steps,
            reasoning="Simple query: single retrieval step",
            estimated_hops=1,
        )

    def _llm_plan(self, query: str) -> RetrievalPlan:
        """Use LLM for query decomposition."""
        prompt = (
            f"Decompose this query into retrieval steps. Return JSON:\n"
            f'{{"steps": [{{"query": "...", "type": "search|compare|aggregate|verify"}}], '
            f'"reasoning": "..."}}\n\nQuery: {query}\n\nJSON:'
        )
        raw = self._plan_fn(prompt)

        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                steps = []
                for i, s in enumerate(data.get("steps", [])):
                    steps.append(PlanStep(
                        step_id=i + 1,
                        query=s.get("query", query),
                        step_type=PlanStepType(s.get("type", "search")),
                    ))
                return RetrievalPlan(
                    original_query=query,
                    steps=steps or [PlanStep(step_id=1, query=query)],
                    reasoning=data.get("reasoning", ""),
                    estimated_hops=len(steps),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        return self._heuristic_plan(query)


class RetrievalEvaluator:
    """Evaluates retrieval quality and decides if more retrieval is needed."""

    def __init__(
        self,
        min_score: float = 0.3,
        min_results: int = 2,
        evaluate_fn: Callable | None = None,
    ):
        self._min_score = min_score
        self._min_results = min_results
        self._evaluate_fn = evaluate_fn

    def evaluate(self, query: str, results: list, round_num: int = 1) -> dict[str, Any]:
        """Evaluate retrieval results and return quality assessment."""
        if not results:
            return {
                "quality": "insufficient",
                "score": 0.0,
                "needs_more": True,
                "reason": "No results retrieved",
            }

        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)
        top_score = max(scores)

        quality = "good" if top_score >= self._min_score and len(results) >= self._min_results else "poor"
        needs_more = quality == "poor" and round_num < 3

        return {
            "quality": quality,
            "score": avg_score,
            "top_score": top_score,
            "result_count": len(results),
            "needs_more": needs_more,
            "reason": "" if quality == "good" else f"Top score {top_score:.3f} below threshold {self._min_score}",
        }


class AgenticPipeline:
    """Full agentic retrieval system: Plan → Retrieve → Evaluate → Critique → Synthesize.

    Orchestrates multi-step retrieval with quality evaluation and optional
    critique loop for complex queries.
    """

    def __init__(
        self,
        retrieve_fn: Callable | None = None,
        generate_fn: Callable | None = None,
        plan_fn: Callable | None = None,
        evaluate_fn: Callable | None = None,
        critique_fn: Callable | None = None,
        max_rounds: int = 3,
        min_confidence: float = 0.5,
    ):
        self._retrieve_fn = retrieve_fn
        self._generate_fn = generate_fn
        self._planner = RetrievalPlanner(plan_fn)
        self._evaluator = RetrievalEvaluator(evaluate_fn=evaluate_fn)
        self._critique_fn = critique_fn
        self._max_rounds = max_rounds
        self._min_confidence = min_confidence

    def run(self, query: str) -> AgenticResult:
        """Execute the full agentic retrieval pipeline."""
        t0 = time.perf_counter()

        # 1. Plan
        plan = self._planner.plan(query)

        # 2. Execute retrieval plan
        all_results = []
        rounds = 0

        for step in plan.steps:
            if step.step_type in (PlanStepType.SEARCH, PlanStepType.VERIFY):
                if self._retrieve_fn:
                    results = self._retrieve_fn(step.query)
                    step.result = results
                    all_results.extend(results)
                    rounds += 1

                    # Evaluate
                    eval_result = self._evaluator.evaluate(step.query, results, rounds)
                    step.score = eval_result.get("score", 0.0)

                    # If poor quality and more rounds allowed, retry with expanded query
                    if eval_result.get("needs_more") and rounds < self._max_rounds:
                        expanded = f"{step.query} (detailed explanation)"
                        retry_results = self._retrieve_fn(expanded)
                        all_results.extend(retry_results)
                        rounds += 1

        # 3. Generate answer
        answer = ""
        if self._generate_fn and all_results:
            answer = self._generate_fn(query, all_results)
        elif not all_results:
            answer = "I could not find sufficient information to answer this question."

        # 4. Critique (optional)
        critique = ""
        confidence = 0.0
        if all_results:
            scores = [r.score for r in all_results if hasattr(r, 'score')]
            confidence = sum(scores) / len(scores) if scores else 0.0

        if self._critique_fn and answer:
            critique = self._critique_fn(query, answer, all_results)

        latency = (time.perf_counter() - t0) * 1000

        return AgenticResult(
            answer=answer,
            plan=plan,
            retrieval_rounds=rounds,
            total_chunks_retrieved=len(all_results),
            critique=critique,
            confidence=confidence,
            latency_ms=latency,
        )
