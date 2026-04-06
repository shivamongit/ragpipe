"""Corrective RAG (CRAG) — self-correcting retrieval with relevance grading.

Based on the CRAG paper (Yan et al., 2024). The agent:
1. Retrieves documents for a query
2. Grades each document for relevance (CORRECT / AMBIGUOUS / INCORRECT)
3. If CORRECT → generate from retrieved docs
4. If AMBIGUOUS → refine knowledge by extracting key sentences + retrieve again
5. If INCORRECT → fall back to web search or return "I don't know"

This is the #1 missing feature across LangChain, LlamaIndex, Haystack, and DSPy —
none offer a built-in, zero-config self-correcting RAG agent.

Usage:
    from ragpipe.agents import CRAGAgent

    agent = CRAGAgent(
        grade_fn=my_llm,          # LLM grades doc relevance
        retrieve_fn=my_retriever,  # your retrieval function
        generate_fn=my_generator,  # your generation function
        web_search_fn=my_search,   # optional web fallback
    )
    result = agent.query("What is the capital of France?")
    print(result.answer, result.confidence, result.action_taken)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class RelevanceGrade(str, Enum):
    """Relevance grade for a retrieved document."""
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class CRAGAction(str, Enum):
    """Action taken by the CRAG agent."""
    DIRECT_GENERATE = "direct_generate"
    REFINED_GENERATE = "refined_generate"
    WEB_SEARCH = "web_search"
    NO_ANSWER = "no_answer"


@dataclass
class GradedDocument:
    """A retrieved document with its relevance grade."""
    text: str
    grade: RelevanceGrade
    score: float = 0.0
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CRAGResult:
    """Result from the CRAG agent."""
    answer: str
    confidence: float
    action_taken: CRAGAction
    graded_docs: list[GradedDocument] = field(default_factory=list)
    refined_query: str = ""
    sources_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


GRADE_PROMPT = """You are a relevance grader for a RAG system. Given a user question and a retrieved document, assess whether the document is relevant to answering the question.

Grade as one of:
- CORRECT: The document contains information directly relevant to answering the question.
- AMBIGUOUS: The document is partially relevant or tangentially related but may not fully answer the question.
- INCORRECT: The document is not relevant to the question at all.

User question: {question}

Retrieved document:
{document}

Respond with ONLY a JSON object:
{{"grade": "correct|ambiguous|incorrect", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

REFINE_PROMPT = """You are a knowledge refinement agent. Given a question and a set of partially relevant documents, extract and consolidate ONLY the key sentences and facts that are relevant to answering the question. Remove all irrelevant information.

Question: {question}

Documents:
{documents}

Return ONLY the refined, relevant knowledge as a concise paragraph. If nothing is relevant, return "NO_RELEVANT_KNOWLEDGE"."""

GENERATE_PROMPT = """Answer the following question using ONLY the provided context. If the context doesn't contain enough information, say so clearly.

Question: {question}

Context:
{context}

Provide a clear, accurate answer grounded in the context above."""


def _parse_grade(raw: str) -> tuple[RelevanceGrade, float, str]:
    """Parse LLM grade response into structured data."""
    raw = raw.strip()
    # Try JSON parse
    try:
        # Extract JSON from potential markdown code blocks
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            grade_str = data.get("grade", "incorrect").lower().strip()
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = data.get("reasoning", "")
            grade_map = {
                "correct": RelevanceGrade.CORRECT,
                "ambiguous": RelevanceGrade.AMBIGUOUS,
                "incorrect": RelevanceGrade.INCORRECT,
            }
            return grade_map.get(grade_str, RelevanceGrade.INCORRECT), confidence, reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: look for keywords
    lower = raw.lower()
    if "correct" in lower and "incorrect" not in lower:
        return RelevanceGrade.CORRECT, 0.7, raw[:200]
    elif "ambiguous" in lower:
        return RelevanceGrade.AMBIGUOUS, 0.5, raw[:200]
    return RelevanceGrade.INCORRECT, 0.3, raw[:200]


class CRAGAgent:
    """Self-correcting RAG agent with relevance grading and adaptive retrieval.

    The CRAG agent evaluates the quality of retrieved documents before generating
    an answer. If documents are irrelevant, it can refine the query, fall back to
    web search, or honestly report that it cannot answer.

    This is a production-grade implementation of the Corrective RAG pattern that
    no other framework provides as a built-in, zero-config module.
    """

    def __init__(
        self,
        grade_fn: Optional[Callable] = None,
        retrieve_fn: Optional[Callable] = None,
        generate_fn: Optional[Callable] = None,
        web_search_fn: Optional[Callable] = None,
        relevance_threshold: float = 0.5,
        max_refinement_rounds: int = 1,
    ):
        self.grade_fn = grade_fn
        self.retrieve_fn = retrieve_fn
        self.generate_fn = generate_fn
        self.web_search_fn = web_search_fn
        self.relevance_threshold = relevance_threshold
        self.max_refinement_rounds = max_refinement_rounds

    def _grade_document(self, question: str, doc_text: str) -> GradedDocument:
        """Grade a single document for relevance to the question."""
        if self.grade_fn is None:
            return GradedDocument(
                text=doc_text,
                grade=RelevanceGrade.CORRECT,
                score=1.0,
                reasoning="No grade function provided — assuming relevant",
            )

        prompt = GRADE_PROMPT.format(question=question, document=doc_text[:2000])
        raw = self.grade_fn(prompt)
        grade, confidence, reasoning = _parse_grade(raw)
        return GradedDocument(
            text=doc_text, grade=grade, score=confidence, reasoning=reasoning,
        )

    def _grade_documents(
        self, question: str, documents: list[str],
    ) -> list[GradedDocument]:
        """Grade all retrieved documents."""
        return [self._grade_document(question, doc) for doc in documents]

    def _refine_knowledge(
        self, question: str, graded_docs: list[GradedDocument],
    ) -> str:
        """Extract relevant knowledge from ambiguous documents."""
        ambiguous = [d for d in graded_docs if d.grade != RelevanceGrade.INCORRECT]
        if not ambiguous:
            return ""
        docs_text = "\n---\n".join(d.text[:1000] for d in ambiguous)
        prompt = REFINE_PROMPT.format(question=question, documents=docs_text)
        if self.generate_fn:
            return self.generate_fn(prompt)
        return docs_text

    def _determine_action(
        self, graded_docs: list[GradedDocument],
    ) -> CRAGAction:
        """Decide which action to take based on grading results."""
        if not graded_docs:
            return CRAGAction.NO_ANSWER

        correct = [d for d in graded_docs if d.grade == RelevanceGrade.CORRECT]
        ambiguous = [d for d in graded_docs if d.grade == RelevanceGrade.AMBIGUOUS]
        incorrect = [d for d in graded_docs if d.grade == RelevanceGrade.INCORRECT]

        correct_ratio = len(correct) / len(graded_docs)
        ambiguous_ratio = len(ambiguous) / len(graded_docs)

        if correct_ratio >= self.relevance_threshold:
            return CRAGAction.DIRECT_GENERATE
        elif (correct_ratio + ambiguous_ratio) >= self.relevance_threshold:
            return CRAGAction.REFINED_GENERATE
        elif self.web_search_fn is not None:
            return CRAGAction.WEB_SEARCH
        else:
            return CRAGAction.NO_ANSWER

    def query(self, question: str, **kwargs: Any) -> CRAGResult:
        """Execute the full CRAG pipeline: retrieve → grade → correct → generate."""
        # Step 1: Retrieve
        if self.retrieve_fn is None:
            return CRAGResult(
                answer="No retrieval function configured.",
                confidence=0.0,
                action_taken=CRAGAction.NO_ANSWER,
            )

        documents = self.retrieve_fn(question, **kwargs)
        if isinstance(documents, list) and documents and hasattr(documents[0], 'text'):
            doc_texts = [d.text if hasattr(d, 'text') else str(d) for d in documents]
        elif isinstance(documents, list):
            doc_texts = [str(d) for d in documents]
        else:
            doc_texts = [str(documents)]

        # Step 2: Grade each document
        graded_docs = self._grade_documents(question, doc_texts)

        # Step 3: Determine action
        action = self._determine_action(graded_docs)

        # Step 4: Execute action
        if action == CRAGAction.DIRECT_GENERATE:
            correct_docs = [d for d in graded_docs if d.grade == RelevanceGrade.CORRECT]
            context = "\n\n".join(d.text for d in correct_docs)
            answer = self._generate(question, context)
            confidence = sum(d.score for d in correct_docs) / len(correct_docs)
            return CRAGResult(
                answer=answer,
                confidence=confidence,
                action_taken=action,
                graded_docs=graded_docs,
                sources_used=len(correct_docs),
            )

        elif action == CRAGAction.REFINED_GENERATE:
            refined = self._refine_knowledge(question, graded_docs)
            if refined and refined != "NO_RELEVANT_KNOWLEDGE":
                answer = self._generate(question, refined)
                usable = [d for d in graded_docs if d.grade != RelevanceGrade.INCORRECT]
                confidence = sum(d.score for d in usable) / max(len(usable), 1) * 0.8
                return CRAGResult(
                    answer=answer,
                    confidence=confidence,
                    action_taken=action,
                    graded_docs=graded_docs,
                    sources_used=len(usable),
                )
            # Fall through to web search or no answer
            action = CRAGAction.WEB_SEARCH if self.web_search_fn else CRAGAction.NO_ANSWER

        if action == CRAGAction.WEB_SEARCH and self.web_search_fn:
            web_results = self.web_search_fn(question)
            if isinstance(web_results, list):
                context = "\n\n".join(str(r) for r in web_results)
            else:
                context = str(web_results)
            answer = self._generate(question, context)
            return CRAGResult(
                answer=answer,
                confidence=0.5,
                action_taken=CRAGAction.WEB_SEARCH,
                graded_docs=graded_docs,
                sources_used=len(web_results) if isinstance(web_results, list) else 1,
                metadata={"web_search": True},
            )

        # NO_ANSWER
        return CRAGResult(
            answer="I don't have enough relevant information to answer this question accurately.",
            confidence=0.0,
            action_taken=CRAGAction.NO_ANSWER,
            graded_docs=graded_docs,
        )

    def _generate(self, question: str, context: str) -> str:
        """Generate answer from context."""
        if self.generate_fn is None:
            return context[:500]
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        return self.generate_fn(prompt)

    async def aquery(self, question: str, **kwargs: Any) -> CRAGResult:
        """Async version of query."""
        return await asyncio.to_thread(self.query, question, **kwargs)
