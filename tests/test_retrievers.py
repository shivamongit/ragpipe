"""Tests for retriever implementations."""

import numpy as np

from ragpipe.core import Chunk
from ragpipe.retrievers.numpy_retriever import NumpyRetriever


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(text=f"chunk {i}", doc_id=f"doc{i}", chunk_index=0)
        for i in range(n)
    ]


def _random_embeddings(n: int, dim: int = 32) -> list[list[float]]:
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    return vecs.tolist()


def test_numpy_retriever_add_and_search():
    retriever = NumpyRetriever()
    chunks = _make_chunks(10)
    embeddings = _random_embeddings(10)

    retriever.add(chunks, embeddings)
    assert retriever.count == 10

    results = retriever.search(embeddings[0], top_k=3)
    assert len(results) == 3
    assert results[0].score >= results[1].score
    assert results[0].chunk.text == "chunk 0"


def test_numpy_retriever_empty():
    retriever = NumpyRetriever()
    results = retriever.search([0.0] * 32, top_k=5)
    assert results == []


def test_numpy_retriever_top_k_larger_than_index():
    retriever = NumpyRetriever()
    chunks = _make_chunks(3)
    embeddings = _random_embeddings(3)
    retriever.add(chunks, embeddings)

    results = retriever.search(embeddings[0], top_k=10)
    assert len(results) == 3
