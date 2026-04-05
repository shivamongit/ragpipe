"""Conversation memory — multi-turn RAG with automatic query contextualization.

Stores conversation history and rewrites follow-up questions to be
self-contained using the chat context. This solves the classic problem
where "What about its performance?" makes no sense without knowing
the previous question was about FAISS.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


CONTEXTUALIZE_PROMPT = """Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that captures the full context needed for retrieval.

If the question is already standalone, return it unchanged.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


class ConversationMemory:
    """Multi-turn conversation memory for RAG pipelines.

    Maintains chat history and automatically contextualizes follow-up
    questions so retrieval works correctly across turns.

    Usage:
        memory = ConversationMemory(contextualize_fn=my_llm_call)

        # Turn 1
        result = memory.query(pipeline, "What is FAISS?")

        # Turn 2 — "it" gets resolved to "FAISS" automatically
        result = memory.query(pipeline, "What about its performance?")
        # Internally rewrites to: "What is FAISS's performance?"

        # Access history
        print(memory.history)
        print(memory.get_context_window(last_n=5))
    """

    def __init__(
        self,
        contextualize_fn: Optional[Callable[[str], str]] = None,
        acontextualize_fn: Optional[Callable] = None,
        max_history: int = 50,
        context_window: int = 10,
    ):
        self._contextualize_fn = contextualize_fn
        self._acontextualize_fn = acontextualize_fn
        self.max_history = max_history
        self.context_window = context_window
        self.history: list[Message] = []

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Add a message to the conversation history."""
        self.history.append(Message(role=role, content=content, metadata=metadata))
        # Trim if over max
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_window(self, last_n: int | None = None) -> list[Message]:
        """Get the last N messages for context."""
        n = last_n or self.context_window
        return self.history[-n:]

    def format_history(self, last_n: int | None = None) -> str:
        """Format recent history as a string for prompt injection."""
        messages = self.get_context_window(last_n)
        if not messages:
            return "(no prior conversation)"
        lines = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            # Truncate long assistant responses for context
            content = msg.content
            if msg.role == "assistant" and len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def contextualize(self, question: str) -> str:
        """Rewrite a follow-up question to be standalone using conversation history.

        If no history or no contextualize_fn, returns the question unchanged.
        """
        if not self.history or not self._contextualize_fn:
            return question

        history_text = self.format_history()
        prompt = CONTEXTUALIZE_PROMPT.format(history=history_text, question=question)

        try:
            result = self._contextualize_fn(prompt)
            return result.strip() if result.strip() else question
        except Exception:
            return question

    async def acontextualize(self, question: str) -> str:
        """Async version of contextualize."""
        if not self.history:
            return question

        if self._acontextualize_fn:
            history_text = self.format_history()
            prompt = CONTEXTUALIZE_PROMPT.format(history=history_text, question=question)
            try:
                result = await self._acontextualize_fn(prompt)
                return result.strip() if result.strip() else question
            except Exception:
                return question
        elif self._contextualize_fn:
            return await asyncio.to_thread(self.contextualize, question)
        else:
            return question

    def query(self, pipeline, question: str, top_k: int | None = None):
        """Run a contextualized query through the pipeline (sync).

        Automatically:
        1. Contextualizes the question using conversation history
        2. Runs the query through the pipeline
        3. Stores the Q&A pair in history
        """
        standalone = self.contextualize(question)

        self.add_message("user", question, standalone_query=standalone)

        result = pipeline.query(standalone, top_k=top_k)

        self.add_message("assistant", result.answer)
        result.metadata["original_question"] = question
        result.metadata["standalone_question"] = standalone

        return result

    async def aquery(self, pipeline, question: str, top_k: int | None = None):
        """Run a contextualized query through the pipeline (async)."""
        standalone = await self.acontextualize(question)

        self.add_message("user", question, standalone_query=standalone)

        result = await pipeline.aquery(standalone, top_k=top_k)

        self.add_message("assistant", result.answer)
        result.metadata["original_question"] = question
        result.metadata["standalone_question"] = standalone

        return result

    def clear(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    @property
    def turn_count(self) -> int:
        """Number of user turns in the conversation."""
        return sum(1 for m in self.history if m.role == "user")
