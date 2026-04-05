"""OpenAI-based generator with source citation prompting."""

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


class OpenAIGenerator(BaseGenerator):
    """Generate cited answers using OpenAI chat completions.

    Supported models (April 2026):
    - gpt-5.4          — flagship, $2.50/$15 per M tokens, 1M ctx, computer use
    - gpt-5.4-pro      — max reasoning, $30/$180 per M tokens
    - gpt-5.3-codex    — agentic coding, $1.75/$14 per M tokens
    - gpt-5-mini       — balanced, $0.25/$2 per M tokens
    - gpt-5-nano       — cheapest, $0.05/$0.40 per M tokens
    - gpt-5            — general purpose, $1.25/$10 per M tokens
    - gpt-4.1          — legacy flagship, $2/$8 per M tokens

    Usage:
        generator = OpenAIGenerator(model="gpt-5.4")
        generator = OpenAIGenerator(model="gpt-5-mini")
    """

    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install 'ragpipe[openai]'")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

        from openai import AsyncOpenAI
        self._aclient = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            label = f"Source {i + 1}"
            parts.append(f"[{label}] (similarity: {r.score:.3f})\n{r.chunk.text}")
        return "\n\n---\n\n".join(parts)

    def generate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {question}",
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        """Native async generate using AsyncOpenAI."""
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)

        response = await self._aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
        """Sync streaming via OpenAI's stream mode."""
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def astream(self, question: str, context: list[RetrievalResult]) -> AsyncIterator[str]:
        """Native async streaming via AsyncOpenAI."""
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)

        response = await self._aclient.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
