"""LiteLLM universal generator — supports 100+ LLM providers via a single interface."""

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


class LiteLLMGenerator(BaseGenerator):
    """Universal generator wrapping 100+ LLM providers via litellm.

    Supports OpenAI, Anthropic, Google Gemini, Mistral, Together AI,
    Groq, Azure OpenAI, AWS Bedrock, Ollama, and more — all via
    the same interface.

    Usage:
        generator = LiteLLMGenerator(model="gpt-4o-mini")
        generator = LiteLLMGenerator(model="claude-sonnet-4-20250514")
        generator = LiteLLMGenerator(model="gemini/gemini-2.0-flash")
        generator = LiteLLMGenerator(model="groq/llama-3.3-70b")
        generator = LiteLLMGenerator(model="ollama/gemma4")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        **kwargs,
    ):
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError("Install litellm: pip install 'ragpipe[litellm]'")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._extra_kwargs = kwargs

    def _build_context(self, results: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(results):
            label = f"Source {i + 1}"
            parts.append(f"[{label}] (score: {r.score:.3f})\n{r.chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _messages(self, question: str, context_text: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
        ]

    def generate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        import litellm

        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)

        response = litellm.completion(
            model=self.model,
            messages=self._messages(question, context_text),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self._extra_kwargs,
        )

        answer = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
            },
        )

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        """Native async generate via litellm.acompletion."""
        import litellm

        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)

        response = await litellm.acompletion(
            model=self.model,
            messages=self._messages(question, context_text),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self._extra_kwargs,
        )

        answer = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
            },
        )

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
        """Sync streaming via litellm."""
        import litellm

        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)

        response = litellm.completion(
            model=self.model,
            messages=self._messages(question, context_text),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **self._extra_kwargs,
        )

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def astream(self, question: str, context: list[RetrievalResult]) -> AsyncIterator[str]:
        """Native async streaming via litellm.acompletion."""
        import litellm

        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)

        response = await litellm.acompletion(
            model=self.model,
            messages=self._messages(question, context_text),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            **self._extra_kwargs,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
