"""FAISS-based vector retriever with persistence."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from ragpipe.core import Chunk, RetrievalResult
from ragpipe.retrievers.base import BaseRetriever


class FaissRetriever(BaseRetriever):
    """High-performance vector retriever backed by FAISS.

    Uses IndexFlatIP with L2-normalized vectors for exact cosine similarity.
    Supports persistence to disk for index reuse across sessions.

    For datasets under 100K vectors, exact search is fast enough.
    For larger datasets, consider switching to IndexIVFFlat or IndexHNSW.
    """

    def __init__(self, dim: int, persist_dir: str | None = None):
        try:
            import faiss
        except ImportError:
            raise ImportError("Install faiss: pip install 'ragpipe[faiss]'")

        self._faiss = faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[dict] = []
        self._persist_dir = Path(persist_dir) if persist_dir else None

        if self._persist_dir:
            self._load()

    def _load(self):
        if not self._persist_dir:
            return
        index_path = self._persist_dir / "faiss.index"
        meta_path = self._persist_dir / "chunks.json"
        if index_path.exists() and meta_path.exists():
            self._index = self._faiss.read_index(str(index_path))
            with open(meta_path, "r") as f:
                self._chunks = json.load(f)

    def _save(self):
        if not self._persist_dir:
            return
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self._persist_dir / "faiss.index"))
        with open(self._persist_dir / "chunks.json", "w") as f:
            json.dump(self._chunks, f)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        self._faiss.normalize_L2(vectors)
        self._index.add(vectors)

        for chunk in chunks:
            self._chunks.append({
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            })

        self._save()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        if self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        self._faiss.normalize_L2(query)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._chunks[idx]
            chunk = Chunk(
                text=meta["text"],
                doc_id=meta["doc_id"],
                chunk_index=meta["chunk_index"],
                metadata=meta["metadata"],
            )
            results.append(RetrievalResult(chunk=chunk, score=float(score)))

        return results

    def delete(self, doc_id: str) -> int:
        """Remove all chunks for a document. Requires index rebuild."""
        keep = [i for i, c in enumerate(self._chunks) if c["doc_id"] != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0

        if not keep:
            self._index = self._faiss.IndexFlatIP(self._dim)
            self._chunks = []
        else:
            vectors = np.array(
                [self._index.reconstruct(i) for i in keep], dtype=np.float32
            )
            self._index = self._faiss.IndexFlatIP(self._dim)
            self._index.add(vectors)
            self._chunks = [self._chunks[i] for i in keep]

        self._save()
        return removed

    @property
    def count(self) -> int:
        return self._index.ntotal
