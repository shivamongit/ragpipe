"""Base generator interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ragpipe.core import RetrievalResult


@dataclass
class GenerationOutput:
    """Raw output from a generator."""

    answer: str
    model: str = ""
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseGenerator(ABC):
    """Abstract base class for LLM generators."""

    @abstractmethod
    def generate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        """Generate an answer given a question and retrieved context."""
        ...

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        """Async generate. Override for native async; default wraps sync in a thread."""
        return await asyncio.to_thread(self.generate, question, context)

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
        """Sync streaming. Override to yield tokens; default yields full answer."""
        result = self.generate(question, context)
        yield result.answer

    async def astream(self, question: str, context: list[RetrievalResult]) -> AsyncIterator[str]:
        """Async streaming. Override for native SSE streaming; default wraps sync."""
        for token in await asyncio.to_thread(lambda: list(self.stream(question, context))):
            yield token
