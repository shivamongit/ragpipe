"""Jina AI embedding provider — 8192 token context window, multilingual."""

from __future__ import annotations

import httpx

from ragpipe.embedders.base import BaseEmbedder

DIMENSIONS = {
    "jina-embeddings-v3": 1024,
    "jina-embeddings-v2-base-en": 768,
    "jina-embeddings-v2-small-en": 512,
    "jina-colbert-v2": 128,
}


class JinaEmbedder(BaseEmbedder):
    """Embed texts using Jina AI's embedding API.

    Jina v3 supports 8192 token context windows — much larger than
    OpenAI's 8191 limit — making it ideal for long chunks.

    Recommended models:
    - jina-embeddings-v3 (1024d, best quality, 8192 tokens)
    - jina-embeddings-v2-base-en (768d, good quality)

    Usage:
        embedder = JinaEmbedder(model="jina-embeddings-v3", api_key="jina_...")
    """

    BASE_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        model: str = "jina-embeddings-v3",
        api_key: str | None = None,
        batch_size: int = 128,
    ):
        import os
        self.model = model
        self._dim = DIMENSIONS.get(model, 1024)
        self.batch_size = batch_size
        self._api_key = api_key or os.environ.get("JINA_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            with httpx.Client() as client:
                resp = client.post(
                    self.BASE_URL,
                    json={"model": self.model, "input": batch},
                    headers=self._headers(),
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                all_embeddings.extend([item["embedding"] for item in data["data"]])

        return all_embeddings

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Native async embed using httpx.AsyncClient."""
        all_embeddings: list[list[float]] = []

        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp = await client.post(
                    self.BASE_URL,
                    json={"model": self.model, "input": batch},
                    headers=self._headers(),
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                all_embeddings.extend([item["embedding"] for item in data["data"]])

        return all_embeddings

    @property
    def dim(self) -> int:
        return self._dim
