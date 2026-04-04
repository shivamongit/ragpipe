"""Ollama embedding provider — fully local, zero-cost."""

from __future__ import annotations

import json
import urllib.request

from ragpipe.embedders.base import BaseEmbedder

DEFAULT_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
}


class OllamaEmbedder(BaseEmbedder):
    """Embed texts using Ollama's local embedding models.

    No API key needed. Runs entirely on your machine.

    Recommended models:
    - nomic-embed-text (768d, excellent quality/speed)
    - mxbai-embed-large (1024d, high quality)
    - snowflake-arctic-embed (1024d, multilingual)
    - bge-m3 (1024d, state-of-the-art multilingual)

    Usage:
        ollama pull nomic-embed-text
        embedder = OllamaEmbedder(model="nomic-embed-text")
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dim: int | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dim = dim or DEFAULT_DIMENSIONS.get(model, 768)

    def _embed_single(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        return data["embeddings"][0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
            return data["embeddings"]
        except Exception:
            return [self._embed_single(t) for t in texts]

    @property
    def dim(self) -> int:
        return self._dim
