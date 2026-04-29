"""Cohere Command generator with async and streaming support."""

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


class CohereGenerator(BaseGenerator):
    """Generate cited answers using Cohere Command models.

    Supported models (April 2026):
    - command-a-03-2025      — flagship Command A, 256K ctx
    - command-r-plus         — best for RAG, $2.50/$10 per M tokens
    - command-r              — balanced, $0.50/$1.50 per M tokens
    - command-r7b            — small/cheap, $0.0375/$0.15 per M tokens

    Cohere has native RAG citations built into the API.

    Usage:
        generator = CohereGenerator(model="command-r-plus")
    """

    def __init__(
        self,
        model: str = "command-r-plus",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        try:
            import cohere
        except ImportError:
            raise ImportError("Install cohere: pip install 'ragpipe[cohere]'")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        key = api_key or os.environ.get("COHERE_API_KEY")
        self._client = cohere.ClientV2(api_key=key) if key else cohere.ClientV2()
        self._aclient = cohere.AsyncClientV2(api_key=key) if key else cohere.AsyncClientV2()

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
        response = self._client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = response.message.content[0].text if response.message.content else ""
        usage = getattr(response, "usage", None)
        billed = getattr(usage, "billed_units", None) if usage else None
        tokens = (
            (getattr(billed, "input_tokens", 0) or 0) + (getattr(billed, "output_tokens", 0) or 0)
        ) if billed else 0
        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=int(tokens),
            metadata={"provider": "cohere"},
        )

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)
        response = await self._aclient.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer = response.message.content[0].text if response.message.content else ""
        usage = getattr(response, "usage", None)
        billed = getattr(usage, "billed_units", None) if usage else None
        tokens = (
            (getattr(billed, "input_tokens", 0) or 0) + (getattr(billed, "output_tokens", 0) or 0)
        ) if billed else 0
        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=int(tokens),
            metadata={"provider": "cohere"},
        )

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)
        for event in self._client.chat_stream(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            if event.type == "content-delta":
                delta_text = event.delta.message.content.text
                if delta_text:
                    yield delta_text

    async def astream(self, question: str, context: list[RetrievalResult]) -> AsyncIterator[str]:
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)
        async for event in self._aclient.chat_stream(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            if event.type == "content-delta":
                delta_text = event.delta.message.content.text
                if delta_text:
                    yield delta_text
