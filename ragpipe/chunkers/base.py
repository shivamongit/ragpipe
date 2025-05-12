"""Base chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragpipe.core import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
        ...
