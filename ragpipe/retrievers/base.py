"""Base retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragpipe.core import Chunk, RetrievalResult


class BaseRetriever(ABC):
    """Abstract base class for vector retrievers."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their embeddings to the index."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        """Search for the most similar chunks."""
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the number of indexed chunks."""
        ...
