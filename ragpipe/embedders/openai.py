"""OpenAI embedding provider."""

from __future__ import annotations

from ragpipe.embedders.base import BaseEmbedder

DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """Embed texts using OpenAI's embedding API with batched requests.

    Supports text-embedding-3-small (1536d), text-embedding-3-large (3072d),
    and text-embedding-ada-002 (1536d). Batches requests in groups of 100
    to stay within API limits.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install 'ragpipe[openai]'")

        self.model = model
        self._dim = DIMENSIONS.get(model, 1536)
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            batch_embs = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embs)

        return all_embeddings

    @property
    def dim(self) -> int:
        return self._dim
