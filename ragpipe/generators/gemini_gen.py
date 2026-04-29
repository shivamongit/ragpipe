"""Google Gemini generator with async and streaming support."""

from __future__ import annotations

import os
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


class GeminiGenerator(BaseGenerator):
    """Generate cited answers using Google Gemini models.

    Supported models (April 2026):
    - gemini-2.5-pro       — flagship, $1.25/$10 per M tokens, 2M ctx
    - gemini-2.5-flash     — fast & cheap, $0.30/$2.50 per M tokens, 1M ctx
    - gemini-2.5-flash-lite — ultra cheap, $0.10/$0.40 per M tokens
    - gemini-2.0-flash     — previous gen, $0.10/$0.40 per M tokens

    Usage:
        generator = GeminiGenerator(model="gemini-2.5-flash")
        generator = GeminiGenerator(model="gemini-2.5-pro", api_key="...")
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install gemini: pip install 'ragpipe[gemini]'")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=key) if key else genai.Client()

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            label = f"Source {i + 1}"
            parts.append(f"[{label}] (score: {r.score:.3f})\n{r.chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, question: str, context: list[RetrievalResult]) -> str:
        context_text = self._build_context(context)
        return f"{self.system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {question}"

    def generate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        prompt = self._build_prompt(question, context)
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )

        answer = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        tokens = getattr(usage, "total_token_count", 0) if usage else 0

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "prompt_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "completion_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
            },
        )

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        prompt = self._build_prompt(question, context)
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        )

        answer = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        tokens = getattr(usage, "total_token_count", 0) if usage else 0

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "prompt_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "completion_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
            },
        )

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        prompt = self._build_prompt(question, context)
        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        ):
            if chunk.text:
                yield chunk.text

    async def astream(self, question: str, context: list[RetrievalResult]) -> AsyncIterator[str]:
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        prompt = self._build_prompt(question, context)
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            },
        ):
            if chunk.text:
                yield chunk.text
