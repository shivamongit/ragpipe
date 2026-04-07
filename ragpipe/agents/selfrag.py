"""Self-Reflective RAG (SelfRAG) — decides when to retrieve and validates its own output.

Based on the SelfRAG paper (Asai et al., 2023). The agent generates reflection tokens
at each step to control the retrieval-augmented generation process:

1. **IsRetrievalNeeded**: Should I retrieve documents? (yes / no / continue)
2. **IsRelevant**: Is each retrieved passage relevant? (relevant / irrelevant)
3. **IsSupported**: Is the response supported by passages? (fully / partially / no)
4. **IsUseful**: Is the response useful to the user? (1–5 scale)

This is a unique differentiator — no other framework provides a built-in self-reflective
RAG agent that can skip retrieval for simple questions, filter irrelevant passages, and
iteratively improve its own output.

Usage:
    from ragpipe.agents import SelfRAGAgent

    agent = SelfRAGAgent(
        retrieve_fn=my_retrieve,
        generate_fn=my_generate,
        reflect_fn=my_llm,  # LLM for reflection tokens
    )
    result = agent.query("What is quantum computing?")
    print(result.answer, result.confidence, result.reflection)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class RetrievalDecision(str, Enum):
    """Whether retrieval is needed for this query."""
    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"
    CONTINUE = "continue"


class RelevanceScore(str, Enum):
    """Whether a retrieved passage is relevant to the query."""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class SupportLevel(str, Enum):
    """How well the generated response is supported by retrieved passages."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


@dataclass
class SelfRAGReflection:
    """Self-reflection tokens from a SelfRAG step."""
    retrieval_needed: RetrievalDecision
    relevance_scores: list[RelevanceScore] = field(default_factory=list)
    support_level: SupportLevel = SupportLevel.NOT_SUPPORTED
    usefulness: int = 3  # 1–5 scale
    reasoning: str = ""


@dataclass
class SelfRAGResult:
    """Result from SelfRAG agent."""
    answer: str
    reflection: SelfRAGReflection
    retrieved_passages: list[str] = field(default_factory=list)
    iterations: int = 1
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Prompts ──────────────────────────────────────────────────────────────────

RETRIEVAL_PROMPT = """You are a retrieval decision agent. Given a question (and optionally a current draft answer), decide whether document retrieval is needed.

Question: {question}
Current answer: {current_answer}

Respond with ONLY a JSON object:
{{"decision": "retrieve|no_retrieve|continue", "reasoning": "brief explanation"}}"""

RELEVANCE_PROMPT = """You are a relevance judge. Given a question and a retrieved passage, decide if the passage is relevant to answering the question.

Question: {question}

Passage:
{passage}

Respond with ONLY a JSON object:
{{"relevance": "relevant|irrelevant", "reasoning": "brief explanation"}}"""

SUPPORT_PROMPT = """You are a support verification agent. Given an answer and supporting passages, determine how well the answer is supported by the passages.

Answer: {answer}

Passages:
{passages}

Respond with ONLY a JSON object:
{{"support": "fully_supported|partially_supported|not_supported", "reasoning": "brief explanation"}}"""

USEFULNESS_PROMPT = """You are a usefulness rater. Given a question and an answer, rate how useful the answer is on a 1–5 scale.

Question: {question}

Answer: {answer}

Respond with ONLY a JSON object:
{{"usefulness": 1-5, "reasoning": "brief explanation"}}"""

GENERATE_PROMPT = """Answer the following question using ONLY the provided context. If the context doesn't contain enough information, say so clearly.

Question: {question}

Context:
{context}

Provide a clear, accurate answer grounded in the context above."""

# Factual question indicators for heuristic retrieval decision
_FACTUAL_INDICATORS = [
    "who ", "what ", "when ", "where ", "which ", "how many", "how much",
    "how long", "how often", "how far", "define ", "name ", "list ",
    "is it true", "did ", "does ", "was ", "were ", "has ", "have ",
]

_OPINION_INDICATORS = [
    "do you think", "in your opinion", "what do you feel",
    "should i", "would you recommend", "personal",
]


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_retrieval_decision(raw: str) -> RetrievalDecision:
    """Parse LLM response into a RetrievalDecision."""
    raw = raw.strip()
    try:
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            decision = data.get("decision", "retrieve").lower().strip()
            mapping = {
                "retrieve": RetrievalDecision.RETRIEVE,
                "no_retrieve": RetrievalDecision.NO_RETRIEVE,
                "continue": RetrievalDecision.CONTINUE,
            }
            return mapping.get(decision, RetrievalDecision.RETRIEVE)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    lower = raw.lower()
    if "no_retrieve" in lower or "no retrieve" in lower:
        return RetrievalDecision.NO_RETRIEVE
    if "continue" in lower:
        return RetrievalDecision.CONTINUE
    return RetrievalDecision.RETRIEVE


def _parse_relevance(raw: str) -> RelevanceScore:
    """Parse LLM response into a RelevanceScore."""
    raw = raw.strip()
    try:
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            rel = data.get("relevance", "irrelevant").lower().strip()
            if rel == "relevant":
                return RelevanceScore.RELEVANT
            return RelevanceScore.IRRELEVANT
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    if "relevant" in raw.lower() and "irrelevant" not in raw.lower():
        return RelevanceScore.RELEVANT
    return RelevanceScore.IRRELEVANT


def _parse_support(raw: str) -> SupportLevel:
    """Parse LLM response into a SupportLevel."""
    raw = raw.strip()
    try:
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            sup = data.get("support", "not_supported").lower().strip()
            mapping = {
                "fully_supported": SupportLevel.FULLY_SUPPORTED,
                "partially_supported": SupportLevel.PARTIALLY_SUPPORTED,
                "not_supported": SupportLevel.NOT_SUPPORTED,
            }
            return mapping.get(sup, SupportLevel.NOT_SUPPORTED)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    lower = raw.lower()
    if "fully" in lower:
        return SupportLevel.FULLY_SUPPORTED
    if "partial" in lower:
        return SupportLevel.PARTIALLY_SUPPORTED
    return SupportLevel.NOT_SUPPORTED


def _parse_usefulness(raw: str) -> int:
    """Parse LLM response into a usefulness score (1–5)."""
    raw = raw.strip()
    try:
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = int(data.get("usefulness", 3))
            return max(1, min(5, score))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: look for a digit
    digits = re.findall(r'[1-5]', raw)
    if digits:
        return int(digits[0])
    return 3


# ── Agent ────────────────────────────────────────────────────────────────────

class SelfRAGAgent:
    """Self-Reflective RAG Agent that decides when to retrieve and validates output.

    Implements the SelfRAG paper approach where the model generates reflection
    tokens at each step to decide retrieval necessity, passage relevance,
    response support, and overall usefulness.

    This is a production-grade implementation that works with or without an LLM:
    when no ``reflect_fn`` is provided, fast heuristic fallbacks handle every
    reflection step so the agent still adds value as a quality gate.
    """

    def __init__(
        self,
        retrieve_fn: Optional[Callable] = None,
        generate_fn: Optional[Callable] = None,
        reflect_fn: Optional[Callable] = None,
        max_iterations: int = 3,
        usefulness_threshold: int = 3,
        support_threshold: str = "partially_supported",
    ):
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.reflect_fn = reflect_fn
        self.max_iterations = max_iterations
        self.usefulness_threshold = usefulness_threshold
        self.support_threshold = SupportLevel(support_threshold)

    # ── Public API ───────────────────────────────────────────────────────

    def query(self, question: str, **kwargs: Any) -> SelfRAGResult:
        """Execute the SelfRAG loop: reflect → retrieve → generate → validate."""
        answer = ""
        passages: list[str] = []
        reflection = SelfRAGReflection(retrieval_needed=RetrievalDecision.RETRIEVE)

        for iteration in range(1, self.max_iterations + 1):
            # Step 1: Should we retrieve?
            decision = self._should_retrieve(question, answer)
            reflection.retrieval_needed = decision

            if decision == RetrievalDecision.RETRIEVE:
                # Step 2: Retrieve
                raw_passages = self._retrieve(question, **kwargs)
                # Step 3: Filter relevant passages
                relevance_scores = self._score_relevance(question, raw_passages)
                reflection.relevance_scores = relevance_scores
                passages = [
                    p for p, s in zip(raw_passages, relevance_scores)
                    if s == RelevanceScore.RELEVANT
                ]
                if not passages:
                    passages = raw_passages[:2]  # keep top results as fallback

            # Step 4: Generate answer
            if passages:
                answer = self._generate_with_context(question, passages)
            elif not answer:
                answer = self._generate_with_context(question, [])

            # Step 5: Check support
            support = self._check_support(answer, passages)
            reflection.support_level = support

            # Step 6: Rate usefulness
            usefulness = self._rate_usefulness(question, answer)
            reflection.usefulness = usefulness

            # Decide whether quality is sufficient
            quality_ok = (
                usefulness >= self.usefulness_threshold
                and self._support_meets_threshold(support)
            )
            if quality_ok or decision == RetrievalDecision.NO_RETRIEVE:
                confidence = self._compute_confidence(reflection, passages)
                reflection.reasoning = f"Converged after {iteration} iteration(s)"
                return SelfRAGResult(
                    answer=answer,
                    reflection=reflection,
                    retrieved_passages=passages,
                    iterations=iteration,
                    confidence=confidence,
                )

        # Exhausted iterations — return best effort
        confidence = self._compute_confidence(reflection, passages)
        reflection.reasoning = f"Max iterations ({self.max_iterations}) reached"
        return SelfRAGResult(
            answer=answer,
            reflection=reflection,
            retrieved_passages=passages,
            iterations=self.max_iterations,
            confidence=confidence,
        )

    async def aquery(self, question: str, **kwargs: Any) -> SelfRAGResult:
        """Async version of query."""
        return await asyncio.to_thread(self.query, question, **kwargs)

    # ── Reflection steps ─────────────────────────────────────────────────

    def _should_retrieve(
        self, question: str, current_answer: str = "",
    ) -> RetrievalDecision:
        """Decide whether retrieval is needed."""
        if self.reflect_fn is None:
            return self._heuristic_should_retrieve(question)
        prompt = RETRIEVAL_PROMPT.format(
            question=question, current_answer=current_answer or "(none)",
        )
        return _parse_retrieval_decision(self.reflect_fn(prompt))

    def _score_relevance(
        self, question: str, passages: list[str],
    ) -> list[RelevanceScore]:
        """Score relevance for each passage."""
        if not passages:
            return []
        if self.reflect_fn is None:
            return [self._heuristic_relevance(question, p) for p in passages]
        scores: list[RelevanceScore] = []
        for passage in passages:
            prompt = RELEVANCE_PROMPT.format(
                question=question, passage=passage[:2000],
            )
            scores.append(_parse_relevance(self.reflect_fn(prompt)))
        return scores

    def _check_support(
        self, answer: str, passages: list[str],
    ) -> SupportLevel:
        """Check how well the answer is supported by passages."""
        if not passages:
            return SupportLevel.NOT_SUPPORTED
        if self.reflect_fn is None:
            return self._heuristic_support(answer, passages)
        combined = "\n---\n".join(p[:1000] for p in passages)
        prompt = SUPPORT_PROMPT.format(answer=answer, passages=combined)
        return _parse_support(self.reflect_fn(prompt))

    def _rate_usefulness(self, question: str, answer: str) -> int:
        """Rate how useful the answer is (1–5)."""
        if not answer:
            return 1
        if self.reflect_fn is None:
            # Heuristic: longer, non-empty answers are more useful
            if len(answer) > 200:
                return 4
            if len(answer) > 50:
                return 3
            return 2
        prompt = USEFULNESS_PROMPT.format(question=question, answer=answer)
        return _parse_usefulness(self.reflect_fn(prompt))

    def _generate_with_context(
        self, question: str, passages: list[str],
    ) -> str:
        """Generate an answer using the given passages as context."""
        context = "\n\n".join(passages) if passages else "(no context available)"
        if self.generate_fn is None:
            return context[:500] if passages else ""
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        return self.generate_fn(prompt)

    # ── Heuristic fallbacks ──────────────────────────────────────────────

    @staticmethod
    def _heuristic_should_retrieve(question: str) -> RetrievalDecision:
        """Rule-based retrieval decision when no LLM is available."""
        q = question.lower().strip()
        if any(ind in q for ind in _OPINION_INDICATORS):
            return RetrievalDecision.NO_RETRIEVE
        if any(ind in q for ind in _FACTUAL_INDICATORS):
            return RetrievalDecision.RETRIEVE
        # Default: retrieve to be safe
        return RetrievalDecision.RETRIEVE

    @staticmethod
    def _heuristic_relevance(question: str, passage: str) -> RelevanceScore:
        """Word-overlap relevance check (>20% overlap → relevant)."""
        q_words = set(question.lower().split())
        p_words = set(passage.lower().split())
        if not q_words:
            return RelevanceScore.IRRELEVANT
        overlap = len(q_words & p_words) / len(q_words)
        return RelevanceScore.RELEVANT if overlap > 0.2 else RelevanceScore.IRRELEVANT

    @staticmethod
    def _heuristic_support(answer: str, passages: list[str]) -> SupportLevel:
        """Word-overlap support check against all passages combined."""
        if not passages:
            return SupportLevel.NOT_SUPPORTED
        a_words = set(answer.lower().split())
        p_words: set[str] = set()
        for p in passages:
            p_words.update(p.lower().split())
        if not a_words:
            return SupportLevel.NOT_SUPPORTED
        overlap = len(a_words & p_words) / len(a_words)
        if overlap > 0.5:
            return SupportLevel.FULLY_SUPPORTED
        if overlap > 0.25:
            return SupportLevel.PARTIALLY_SUPPORTED
        return SupportLevel.NOT_SUPPORTED

    # ── Internal helpers ─────────────────────────────────────────────────

    def _retrieve(self, question: str, **kwargs: Any) -> list[str]:
        """Invoke the retrieve function and normalise results to strings."""
        if self.retrieve_fn is None:
            return []
        results = self.retrieve_fn(question, **kwargs)
        if isinstance(results, list):
            return [
                r.text if hasattr(r, "text") else str(r) for r in results
            ]
        return [str(results)]

    def _support_meets_threshold(self, support: SupportLevel) -> bool:
        """Check whether support level meets or exceeds the configured threshold."""
        order = [
            SupportLevel.NOT_SUPPORTED,
            SupportLevel.PARTIALLY_SUPPORTED,
            SupportLevel.FULLY_SUPPORTED,
        ]
        return order.index(support) >= order.index(self.support_threshold)

    @staticmethod
    def _compute_confidence(
        reflection: SelfRAGReflection, passages: list[str],
    ) -> float:
        """Derive a 0-1 confidence from the reflection tokens."""
        score = 0.0

        # Support contributes 40%
        support_map = {
            SupportLevel.FULLY_SUPPORTED: 0.4,
            SupportLevel.PARTIALLY_SUPPORTED: 0.2,
            SupportLevel.NOT_SUPPORTED: 0.0,
        }
        score += support_map.get(reflection.support_level, 0.0)

        # Usefulness contributes 40% (normalised from 1-5 → 0-1)
        score += (reflection.usefulness - 1) / 4 * 0.4

        # Passage count contributes 20%
        if passages:
            relevant = sum(
                1 for s in reflection.relevance_scores
                if s == RelevanceScore.RELEVANT
            )
            score += min(relevant / 3, 1.0) * 0.2

        return min(score, 1.0)
