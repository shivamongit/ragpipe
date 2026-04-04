"""Query expansion strategies — HyDE, Multi-Query, Step-Back prompting.

These transform the raw user query before retrieval to improve recall.
Each expander returns one or more query strings that can be embedded
and searched independently, then results are deduplicated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseExpander(ABC):
    """Abstract base for query expansion strategies."""

    @abstractmethod
    def expand(self, query: str) -> list[str]:
        """Return one or more expanded queries."""
        ...


class HyDEExpander(BaseExpander):
    """Hypothetical Document Embeddings — generate a hypothetical answer,
    then search for documents similar to that answer.

    The hypothesis often matches document language better than the question.
    Requires a text generation function (LLM call).

    Parameters:
        generate_fn: Callable(prompt: str) -> str
    """

    PROMPT = (
        "Write a short, factual paragraph that directly answers the following question. "
        "Do not include any preamble or hedging — just answer as if you were writing a "
        "reference document.\n\nQuestion: {query}"
    )

    def __init__(self, generate_fn):
        self._generate = generate_fn

    def expand(self, query: str) -> list[str]:
        prompt = self.PROMPT.format(query=query)
        hypothesis = self._generate(prompt)
        return [query, hypothesis.strip()]


class MultiQueryExpander(BaseExpander):
    """Generate multiple query reformulations for broader recall.

    The LLM creates N diverse reformulations of the original query.
    Each reformulation is searched independently; results are merged
    and deduplicated by chunk ID.

    Parameters:
        generate_fn: Callable(prompt: str) -> str
        n_queries: Number of alternative queries to generate (default 3)
    """

    PROMPT = (
        "Generate {n} different search queries that would help answer this question. "
        "Each query should approach the question from a different angle or use different "
        "terminology. Return ONLY the queries, one per line, no numbering.\n\n"
        "Original question: {query}"
    )

    def __init__(self, generate_fn, n_queries: int = 3):
        self._generate = generate_fn
        self.n_queries = n_queries

    def expand(self, query: str) -> list[str]:
        prompt = self.PROMPT.format(n=self.n_queries, query=query)
        response = self._generate(prompt)
        alternatives = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return [query] + alternatives[: self.n_queries]


class StepBackExpander(BaseExpander):
    """Step-back prompting — ask a broader question first, then combine results.

    For specific questions, retrieval often fails because the exact answer
    isn't in any single chunk. A broader question retrieves supporting context
    that helps answer the specific question.

    Parameters:
        generate_fn: Callable(prompt: str) -> str
    """

    PROMPT = (
        "Given this specific question, what is a more general question that would "
        "help retrieve background information needed to answer it?\n\n"
        "Specific question: {query}\n\n"
        "General question (respond with ONLY the question):"
    )

    def __init__(self, generate_fn):
        self._generate = generate_fn

    def expand(self, query: str) -> list[str]:
        prompt = self.PROMPT.format(query=query)
        broader = self._generate(prompt).strip()
        return [query, broader]
