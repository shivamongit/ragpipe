"""Mistral AI generator with async and streaming support."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

from ragpipe.core import RetrievalResult
from ragpipe.generators.base import BaseGenerator, GenerationOutput

SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer questions using ONLY the provided context chunks.

Rules:
1. Base your answer exclusively on the provided context
2. Cite sources using [Source N] notation for each claim
3. If the context doesn't contain enough information, say so clearly
4. Be concise and factual — no speculation
5. Synthesize multiple sources when they agree
6. Preserve technical terminology from the source documents"""


class MistralGenerator(BaseGenerator):
    """Generate cited answers using Mistral AI models.

    Supported models (April 2026):
    - mistral-large-2411     — flagship, $2/$6 per M tokens, 128K ctx
    - mistral-medium-2505    — balanced, $0.40/$2 per M tokens
    - mistral-small-2503     — efficient, $0.10/$0.30 per M tokens
    - codestral-2501         — coding specialist
    - magistral-medium       — reasoning
    - pixtral-large-2411     — multimodal

    Usage:
        generator = MistralGenerator(model="mistral-large-2411")
    """

    def __init__(
        self,
        model: str = "mistral-large-2411",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        try:
            from mistralai import Mistral
        except ImportError:
            raise ImportError("Install mistralai: pip install 'ragpipe[mistral]'")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise ValueError("MISTRAL_API_KEY required (set env var or pass api_key=)")
        self._client = Mistral(api_key=key)

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
        response = self._client.chat.complete(
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
                "provider": "mistral",
            },
        )

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)
        response = await self._client.chat.complete_async(
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
                "provider": "mistral",
            },
        )

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)
        for event in self._client.chat.stream(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            delta = event.data.choices[0].delta if event.data.choices else None
            if delta and delta.content:
                yield delta.content

    async def astream(self, question: str, context: list[RetrievalResult]) -> AsyncIterator[str]:
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)
        async for event in await self._client.chat.stream_async(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            delta = event.data.choices[0].delta if event.data.choices else None
            if delta and delta.content:
                yield delta.content
