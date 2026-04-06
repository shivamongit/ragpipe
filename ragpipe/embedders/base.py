"""Base embedder interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into vectors."""
        ...

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        on_progress: callable | None = None,
    ) -> list[list[float]]:
        """Embed texts in batches with optional progress callback.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch.
            on_progress: Optional callback(completed, total) called after each batch.

        Returns:
            List of embedding vectors.
        """
        all_embeddings: list[list[float]] = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embed(batch)
            all_embeddings.extend(embeddings)
            if on_progress:
                on_progress(min(i + batch_size, total), total)
        return all_embeddings

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Async embed. Override for native async; default wraps sync in a thread."""
        return await asyncio.to_thread(self.embed, texts)

    async def aembed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        max_concurrency: int = 4,
        on_progress: callable | None = None,
    ) -> list[list[float]]:
        """Async batch embedding with configurable concurrency.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch.
            max_concurrency: Max concurrent embedding requests.
            on_progress: Optional callback(completed, total).

        Returns:
            List of embedding vectors in original order.
        """
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        results: list[list[list[float]]] = [[] for _ in batches]
        semaphore = asyncio.Semaphore(max_concurrency)
        completed = 0
        total = len(texts)

        async def _embed_batch(idx: int, batch: list[str]):
            nonlocal completed
            async with semaphore:
                embeddings = await self.aembed(batch)
                results[idx] = embeddings
                completed += len(batch)
                if on_progress:
                    on_progress(min(completed, total), total)

        tasks = [_embed_batch(i, b) for i, b in enumerate(batches)]
        await asyncio.gather(*tasks)

        all_embeddings: list[list[float]] = []
        for batch_result in results:
            all_embeddings.extend(batch_result)
        return all_embeddings

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the embedding dimension."""
        ...
