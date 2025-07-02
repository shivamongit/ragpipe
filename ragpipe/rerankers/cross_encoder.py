"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

from ragpipe.core import RetrievalResult
from ragpipe.rerankers.base import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Rerank results using a cross-encoder model.

    Cross-encoders process (query, passage) pairs jointly, producing
    more accurate relevance scores than bi-encoder similarity.
    Significantly improves precision at the cost of latency.

    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
    - BAAI/bge-reranker-large (high quality)
    """

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install 'ragpipe[sentence-transformers]'"
            )

        self._model = CrossEncoder(model)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 3
    ) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)

        for result, score in zip(results, scores):
            result.score = float(score)

        ranked = sorted(results, key=lambda r: r.score, reverse=True)
        return ranked[:top_k]
