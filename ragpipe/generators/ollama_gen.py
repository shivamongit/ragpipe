"""Ollama-based generator — fully local, zero-cost LLM generation."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import AsyncIterator, Iterator

import httpx

from ragpipe.core import RetrievalResult
from ragpipe.generators.base import BaseGenerator, GenerationOutput

SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer questions using ONLY the provided context chunks. Follow these rules strictly:

1. Base your answer exclusively on the provided context
2. Cite sources using [Source N] notation for each claim
3. If the context doesn't contain enough information, say so clearly
4. Be concise and factual — no speculation
5. If multiple sources agree, synthesize them into a coherent answer
6. Preserve technical terminology from the source documents"""


class OllamaGenerator(BaseGenerator):
    """Generate cited answers using local Ollama models.

    No API key, no cost, no data leaving your machine.

    Popular local models (April 2026):
    - gemma4:27b         — Google, best quality/speed for RAG
    - qwen3.5:32b        — Alibaba, hybrid thinking, strong reasoning
    - llama4:scout       — Meta, 10M context, MoE multimodal
    - llama4:maverick    — Meta, frontier quality, MoE
    - deepseek-v3.2      — DeepSeek, MIT license, GPT-5 class
    - deepseek-r1        — DeepSeek, reasoning specialist
    - nemotron3:super    — Nvidia, Mamba-Transformer MoE
    - mistral-small:3.2  — Mistral, compact and fast
    - phi-4              — Microsoft, compact, fast
    - qwen3-coder        — Alibaba, code specialist

    Usage:
        ollama pull gemma4:27b
        generator = OllamaGenerator(model="gemma4:27b")
    """

    def __init__(
        self,
        model: str = "gemma4",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_ctx: int = 8192,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.system_prompt = system_prompt or SYSTEM_PROMPT

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

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {question}",
                },
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())

        answer = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0),
                "eval_duration_ns": data.get("eval_duration", 0),
                "total_duration_ns": data.get("total_duration", 0),
            },
        )

    async def agenerate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        """Native async generate using httpx."""
        if not context:
            return GenerationOutput(
                answer="No relevant context found. Please provide documents first.",
                model=self.model,
            )

        context_text = self._build_context(context)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
                    ],
                    "stream": False,
                    "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()

        answer = data.get("message", {}).get("content", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        return GenerationOutput(
            answer=answer,
            model=self.model,
            tokens_used=tokens,
            metadata={
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0),
            },
        )

    def stream(self, question: str, context: list[RetrievalResult]) -> Iterator[str]:
        """Sync streaming via Ollama's stream mode."""
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
            "stream": True,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line.decode())
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content

    async def astream(self, question: str, context: list[RetrievalResult]) -> AsyncIterator[str]:
        """Native async streaming via httpx."""
        if not context:
            yield "No relevant context found. Please provide documents first."
            return

        context_text = self._build_context(context)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
                    ],
                    "stream": True,
                    "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
                },
                timeout=120.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
