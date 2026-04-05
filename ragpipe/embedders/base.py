"""Base embedder interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        ...

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Async embed. Override for native async; default wraps sync in a thread."""
        return await asyncio.to_thread(self.embed, texts)

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the embedding dimension."""
        ...
