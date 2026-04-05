"""Anthropic Claude generator with async and streaming support."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from ragpipe.core import RetrievalResult
from ragpipe.generators.base import BaseGenerator, GenerationOutput

SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer questions using ONLY the provided context chunks. Follow these rules strictly:

1. Base your answer exclusively on the provided context
2. Cite sources using [Source N] notation for each claim
3. If the context doesn't contain enough information, say so clearly
4. Be concise and factual — no speculation
5. If multiple sources agree, synthesize them into a coherent answer
6. Preserve technical terminology from the source documents"""


class AnthropicGenerator(BaseGenerator):
    """Generate cited answers using Anthropic Claude models.

    Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus, etc.

    Usage:
        generator = AnthropicGenerator(model="claude-sonnet-4-20250514")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install 'ragpipe[anthropic]'")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self._aclient = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            label = f"Source {i + 1}"
            parts.append(f"[{label}] (score: {r.score:.3f})\n{r.chunk.text}")
        return "\n\n---\n\n".join(parts)

    def generate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)

        response = self._client.messages.create(
            model=self.model,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.content[0].text if response.content else ""
        tokens = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        """Native async generate using AsyncAnthropic."""
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)

        response = await self._aclient.messages.create(
            model=self.model,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.content[0].text if response.content else ""
        tokens = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
        """Sync streaming via Anthropic's stream mode."""
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)

        with self._client.messages.stream(
            model=self.model,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text

    async def astream(self, question: str, context: list[RetrievalResult]) -> AsyncIterator[str]:
        """Native async streaming via AsyncAnthropic."""
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)

        async with self._aclient.messages.stream(
            model=self.model,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text
