"""LLM-as-Judge evaluation — score faithfulness, relevance, and completeness using an LLM.

Replaces lexical-only metrics with LLM-judged quality scores (0–5 scale with reasoning).
This is the RAGAS-style evaluation approach used in production RAG systems.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class JudgmentResult:
    """Result of an LLM judge evaluation."""
    score: float  # 0-5 scale
    reasoning: str = ""
    dimension: str = ""  # faithfulness, relevance, completeness
    metadata: dict[str, Any] = field(default_factory=dict)


FAITHFULNESS_PROMPT = """You are an expert judge evaluating the faithfulness of a RAG system's answer.

Faithfulness measures whether the answer is fully supported by the provided context chunks. An answer is faithful if every claim it makes can be traced back to the context.

Context chunks:
{context}

Question: {question}
Answer: {answer}

Score the faithfulness from 0 to 5:
- 5: Every claim is directly supported by the context
- 4: Almost all claims are supported, minor unsupported details
- 3: Most claims are supported, some unsupported assertions
- 2: Several claims lack support from the context
- 1: Most claims are unsupported or fabricated
- 0: The answer is completely made up / contradicts the context

Respond with ONLY a JSON object:
{{"score": <0-5>, "reasoning": "<brief explanation>"}}"""


RELEVANCE_PROMPT = """You are an expert judge evaluating the relevance of a RAG system's answer.

Relevance measures whether the answer actually addresses the user's question.

Question: {question}
Answer: {answer}

Score the relevance from 0 to 5:
- 5: Directly and completely answers the question
- 4: Answers the question well with minor tangents
- 3: Partially answers the question
- 2: Somewhat related but doesn't really answer the question
- 1: Barely related to the question
- 0: Completely off-topic or non-responsive

Respond with ONLY a JSON object:
{{"score": <0-5>, "reasoning": "<brief explanation>"}}"""


COMPLETENESS_PROMPT = """You are an expert judge evaluating the completeness of a RAG system's answer.

Completeness measures whether the answer covers all aspects of the question given the available context.

Context chunks:
{context}

Question: {question}
Answer: {answer}

Score the completeness from 0 to 5:
- 5: Covers every relevant aspect from the context
- 4: Covers most aspects, misses minor details
- 3: Covers the main points but misses significant details
- 2: Covers only some aspects
- 1: Very superficial, misses most relevant information
- 0: Empty or completely incomplete

Respond with ONLY a JSON object:
{{"score": <0-5>, "reasoning": "<brief explanation>"}}"""


def _parse_judgment(text: str, dimension: str) -> JudgmentResult:
    """Parse LLM judgment response into a JudgmentResult."""
    text = text.strip()
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            score = float(data.get("score", 0))
            score = max(0.0, min(5.0, score))
            return JudgmentResult(
                score=score,
                reasoning=data.get("reasoning", ""),
                dimension=dimension,
            )
        except (json.JSONDecodeError, ValueError):
            pass

    return JudgmentResult(score=0.0, reasoning="Failed to parse judgment", dimension=dimension)


class LLMJudge:
    """Evaluate RAG answers using an LLM as judge.

    Scores three dimensions:
    - Faithfulness: Is the answer grounded in the context?
    - Relevance: Does the answer address the question?
    - Completeness: Does the answer cover all relevant aspects?

    Usage:
        judge = LLMJudge(judge_fn=my_llm_call)
        scores = judge.evaluate(
            question="What is FAISS?",
            answer="FAISS is a library...",
            context_texts=["FAISS is developed by Meta..."]
        )
        print(scores["faithfulness"].score)  # 0-5
        print(scores["overall"])             # weighted average
    """

    def __init__(
        self,
        judge_fn: Optional[Callable[[str], str]] = None,
        ajudge_fn: Optional[Callable] = None,
        weights: dict[str, float] | None = None,
    ):
        self._judge_fn = judge_fn
        self._ajudge_fn = ajudge_fn
        self.weights = weights or {
            "faithfulness": 0.4,
            "relevance": 0.35,
            "completeness": 0.25,
        }

    def _format_context(self, context_texts: list[str]) -> str:
        parts = []
        for i, text in enumerate(context_texts):
            parts.append(f"[Chunk {i+1}]\n{text}")
        return "\n\n---\n\n".join(parts)

    def judge_faithfulness(
        self, question: str, answer: str, context_texts: list[str]
    ) -> JudgmentResult:
        """Judge faithfulness (sync)."""
        if not self._judge_fn:
            return JudgmentResult(score=0, reasoning="No judge_fn provided", dimension="faithfulness")

        context = self._format_context(context_texts)
        prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
        response = self._judge_fn(prompt)
        return _parse_judgment(response, "faithfulness")

    def judge_relevance(self, question: str, answer: str) -> JudgmentResult:
        """Judge relevance (sync)."""
        if not self._judge_fn:
            return JudgmentResult(score=0, reasoning="No judge_fn provided", dimension="relevance")

        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        response = self._judge_fn(prompt)
        return _parse_judgment(response, "relevance")

    def judge_completeness(
        self, question: str, answer: str, context_texts: list[str]
    ) -> JudgmentResult:
        """Judge completeness (sync)."""
        if not self._judge_fn:
            return JudgmentResult(score=0, reasoning="No judge_fn provided", dimension="completeness")

        context = self._format_context(context_texts)
        prompt = COMPLETENESS_PROMPT.format(context=context, question=question, answer=answer)
        response = self._judge_fn(prompt)
        return _parse_judgment(response, "completeness")

    def evaluate(
        self,
        question: str,
        answer: str,
        context_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run all three judges and return scores + weighted overall score."""
        context_texts = context_texts or []

        faithfulness = self.judge_faithfulness(question, answer, context_texts)
        relevance = self.judge_relevance(question, answer)
        completeness = self.judge_completeness(question, answer, context_texts)

        overall = (
            faithfulness.score * self.weights["faithfulness"]
            + relevance.score * self.weights["relevance"]
            + completeness.score * self.weights["completeness"]
        )

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "overall": round(overall, 2),
        }

    async def aevaluate(
        self,
        question: str,
        answer: str,
        context_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Async evaluation — runs all three judges in parallel."""
        context_texts = context_texts or []

        if self._ajudge_fn:
            context = self._format_context(context_texts)

            async def _afaithfulness():
                prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
                resp = await self._ajudge_fn(prompt)
                return _parse_judgment(resp, "faithfulness")

            async def _arelevance():
                prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
                resp = await self._ajudge_fn(prompt)
                return _parse_judgment(resp, "relevance")

            async def _acompleteness():
                prompt = COMPLETENESS_PROMPT.format(context=context, question=question, answer=answer)
                resp = await self._ajudge_fn(prompt)
                return _parse_judgment(resp, "completeness")

            faithfulness, relevance, completeness = await asyncio.gather(
                _afaithfulness(), _arelevance(), _acompleteness()
            )
        else:
            # Fallback: run sync in thread
            result = await asyncio.to_thread(self.evaluate, question, answer, context_texts)
            return result

        overall = (
            faithfulness.score * self.weights["faithfulness"]
            + relevance.score * self.weights["relevance"]
            + completeness.score * self.weights["completeness"]
        )

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "overall": round(overall, 2),
        }
