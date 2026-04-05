"""Voyage AI embedding provider — high-quality embeddings that outperform OpenAI on many benchmarks."""

from __future__ import annotations

from ragpipe.embedders.base import BaseEmbedder

DIMENSIONS = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-large-2": 1536,
    "voyage-2": 1024,
}


class VoyageEmbedder(BaseEmbedder):
    """Embed texts using Voyage AI's embedding API.

    Voyage embeddings consistently outperform OpenAI on MTEB benchmarks,
    especially for retrieval and code tasks.

    Recommended models:
    - voyage-3 (1024d, best general-purpose)
    - voyage-code-3 (1024d, optimized for code)
    - voyage-3-lite (512d, fast and cheap)

    Usage:
        embedder = VoyageEmbedder(model="voyage-3")
    """

    def __init__(
        self,
        model: str = "voyage-3",
        api_key: str | None = None,
        batch_size: int = 128,
    ):
        try:
            import voyageai
        except ImportError:
            raise ImportError("Install voyageai: pip install 'ragpipe[voyage]'")

        self.model = model
        self._dim = DIMENSIONS.get(model, 1024)
        self.batch_size = batch_size
        self._client = voyageai.Client(api_key=api_key) if api_key else voyageai.Client()

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self._client.embed(batch, model=self.model)
            all_embeddings.extend(result.embeddings)

        return all_embeddings

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Native async embed using httpx."""
        import httpx
        import os

        api_key = os.environ.get("VOYAGE_API_KEY", "")
        all_embeddings: list[list[float]] = []

        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp = await client.post(
                    "https://api.voyageai.com/v1/embeddings",
                    json={"model": self.model, "input": batch},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                all_embeddings.extend([item["embedding"] for item in data["data"]])

        return all_embeddings

    @property
    def dim(self) -> int:
        return self._dim
