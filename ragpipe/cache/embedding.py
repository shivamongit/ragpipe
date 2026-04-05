"""Embedding cache — LRU cache for embed() calls.

Prevents re-embedding identical texts. Wraps any embedder
to transparently cache results by text content hash.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


class EmbeddingCache:
    """LRU cache for embedding results keyed by text content hash.

    Usage:
        cache = EmbeddingCache(max_size=10000)

        # Check cache before calling embedder
        cached = cache.get(texts)
        if cached is not None:
            return cached

        embeddings = embedder.embed(texts)
        cache.put(texts, embeddings)
        return embeddings

    Or wrap an embedder:
        cached_embedder = cache.wrap(embedder)
        embeddings = cached_embedder.embed(texts)  # auto-cached
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_single(self, text: str) -> list[float] | None:
        """Get cached embedding for a single text."""
        key = self._hash_text(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put_single(self, text: str, embedding: list[float]) -> None:
        """Cache a single text→embedding pair."""
        key = self._hash_text(text)
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def get(self, texts: list[str]) -> list[list[float]] | None:
        """Get cached embeddings for a batch. Returns None if ANY text is missing."""
        results = []
        for text in texts:
            emb = self.get_single(text)
            if emb is None:
                return None
            results.append(emb)
        return results

    def put(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Cache a batch of text→embedding pairs."""
        for text, emb in zip(texts, embeddings):
            self.put_single(text, emb)

    def get_partial(self, texts: list[str]) -> tuple[list[list[float] | None], list[int]]:
        """Get cached embeddings, returning None for misses.

        Returns:
            (results, missing_indices): results[i] is embedding or None,
            missing_indices lists which positions need embedding.
        """
        results: list[list[float] | None] = []
        missing: list[int] = []
        for i, text in enumerate(texts):
            emb = self.get_single(text)
            results.append(emb)
            if emb is None:
                missing.append(i)
        return results, missing

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
        }
