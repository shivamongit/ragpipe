"""Hybrid retriever — combines dense vector + sparse BM25 with Reciprocal Rank Fusion."""

from __future__ import annotations

from ragpipe.core import Chunk, RetrievalResult
from ragpipe.retrievers.base import BaseRetriever


class HybridRetriever(BaseRetriever):
    """Fuse dense (vector) and sparse (BM25) retrieval using Reciprocal Rank Fusion.

    RRF merges ranked lists without requiring score normalization.
    A document ranked #1 in dense and #3 in sparse scores higher
    than one ranked #2 in both.

    Parameters:
        dense_retriever: Any BaseRetriever for vector search (FAISS, NumPy)
        sparse_retriever: BM25Retriever for keyword search
        dense_weight: Weight for dense retrieval scores (default 0.6)
        sparse_weight: Weight for sparse retrieval scores (default 0.4)
        rrf_k: RRF constant — higher values flatten rank differences (default 60)
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks to both dense and sparse indexes."""
        self._dense.add(chunks, embeddings)
        self._sparse.add(chunks, embeddings)

    def _rrf_score(self, rank: int, weight: float) -> float:
        return weight / (self.rrf_k + rank)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        query_text: str | None = None,
    ) -> list[RetrievalResult]:
        """Search using both dense and sparse retrieval, fused with RRF.

        Args:
            query_embedding: Dense vector for semantic search
            top_k: Number of results to return
            query_text: Raw query string for BM25 (required for sparse search)
        """
        fetch_k = top_k * 3

        dense_results = self._dense.search(query_embedding, top_k=fetch_k)

        sparse_results = []
        if query_text:
            sparse_results = self._sparse.search_text(query_text, top_k=fetch_k)

        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, r in enumerate(dense_results, start=1):
            cid = r.chunk.id
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + self._rrf_score(rank, self.dense_weight)
            chunk_map[cid] = r.chunk

        for rank, r in enumerate(sparse_results, start=1):
            cid = r.chunk.id
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + self._rrf_score(rank, self.sparse_weight)
            if cid not in chunk_map:
                chunk_map[cid] = r.chunk

        sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)

        results = []
        for cid in sorted_ids[:top_k]:
            results.append(
                RetrievalResult(chunk=chunk_map[cid], score=chunk_scores[cid])
            )

        return results

    @property
    def count(self) -> int:
        return self._dense.count
