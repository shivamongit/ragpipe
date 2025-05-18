"""Base embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the embedding dimension."""
        ...
