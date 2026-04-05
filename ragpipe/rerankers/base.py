"""Base reranker interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragpipe.core import RetrievalResult


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 3
    ) -> list[RetrievalResult]:
        """Rerank retrieval results by relevance to the query."""
        ...

    async def arerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 3
    ) -> list[RetrievalResult]:
        """Async rerank. Override for native async; default wraps sync in a thread."""
        return await asyncio.to_thread(self.rerank, query, results, top_k)
