"""OpenAI-based generator with source citation prompting."""

from __future__ import annotations

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

    Builds a structured context window from retrieved chunks,
    prompts the model to cite sources, and returns the answer
    with token usage metadata.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
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
