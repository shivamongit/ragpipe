"""Sentence Transformers embedding provider — runs fully local."""

from __future__ import annotations

from ragpipe.embedders.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embed texts using sentence-transformers models locally.

    No API calls, no cost. Runs on CPU or GPU. Recommended models:
    - all-MiniLM-L6-v2 (384d, fast)
    - all-mpnet-base-v2 (768d, balanced)
    - BAAI/bge-large-en-v1.5 (1024d, high quality)
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install 'ragpipe[sentence-transformers]'"
            )

        self._model = SentenceTransformer(model, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    @property
    def dim(self) -> int:
        return self._dim
