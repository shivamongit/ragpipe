"""Pure NumPy retriever — zero dependencies beyond numpy."""

from __future__ import annotations

import numpy as np

from ragpipe.core import Chunk, RetrievalResult
from ragpipe.retrievers.base import BaseRetriever


class NumpyRetriever(BaseRetriever):
    """Cosine similarity retriever using only NumPy.

    No FAISS dependency required. Suitable for small-to-medium datasets
    (< 50K chunks) or environments where installing FAISS is impractical.
    """

    def __init__(self):
        self._vectors: np.ndarray | None = None
        self._chunks: list[dict] = []

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        new_vectors = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        new_vectors = new_vectors / norms

        if self._vectors is None:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        for chunk in chunks:
            self._chunks.append({
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            })

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        scores = self._vectors @ query
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            meta = self._chunks[idx]
            chunk = Chunk(
                text=meta["text"],
                doc_id=meta["doc_id"],
                chunk_index=meta["chunk_index"],
                metadata=meta["metadata"],
            )
            results.append(RetrievalResult(chunk=chunk, score=float(scores[idx])))

        return results

    @property
    def count(self) -> int:
        return len(self._chunks)
