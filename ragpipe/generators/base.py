"""Base generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
