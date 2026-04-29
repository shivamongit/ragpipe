"""Groq generator (OpenAI-compatible API, ultra-fast inference)."""

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


class GroqGenerator(BaseGenerator):
    """Generate cited answers using Groq (ultra-fast LPU inference).

    Supported models (April 2026):
    - llama-3.3-70b-versatile      — best balance, ~275 tok/s
    - llama-3.1-8b-instant         — fastest, ~750 tok/s
    - mixtral-8x7b-32768           — long context
    - qwen-3-32b                   — strong reasoning
    - deepseek-r1-distill-llama-70b — reasoning model
    - kimi-k2                      — agentic

    Free tier available. Get key at https://console.groq.com/

    Usage:
        generator = GroqGenerator(model="llama-3.3-70b-versatile")
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai (used for Groq's OpenAI-compatible API): pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY required (set env var or pass api_key=)")

        base_url = "https://api.groq.com/openai/v1"
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._aclient = AsyncOpenAI(api_key=key, base_url=base_url)

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
        response = self._client.chat.completions.create(
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
                "provider": "groq",
            },
        )

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
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
                "provider": "groq",
            },
        )

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
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
