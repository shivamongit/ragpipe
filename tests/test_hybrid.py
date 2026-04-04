"""Tests for hybrid retriever with RRF."""

import numpy as np

from ragpipe.core import Chunk
from ragpipe.retrievers.numpy_retriever import NumpyRetriever
from ragpipe.retrievers.bm25_retriever import BM25Retriever
from ragpipe.retrievers.hybrid_retriever import HybridRetriever


def _make_chunks():
    return [
        Chunk(text="Python is a programming language for data science", doc_id="doc1", chunk_index=0),
        Chunk(text="JavaScript powers modern web frontend applications", doc_id="doc2", chunk_index=0),
        Chunk(text="FAISS enables fast similarity search over vectors", doc_id="doc3", chunk_index=0),
        Chunk(text="BM25 ranks documents by keyword relevance scores", doc_id="doc4", chunk_index=0),
    ]


def _random_embeddings(n, dim=32):
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32).tolist()


def test_hybrid_add_and_search():
    dense = NumpyRetriever()
    sparse = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    chunks = _make_chunks()
    embeddings = _random_embeddings(4)
    hybrid.add(chunks, embeddings)

    assert hybrid.count == 4

    results = hybrid.search(embeddings[0], top_k=3, query_text="Python data science")
    assert len(results) > 0
    assert len(results) <= 3


def test_hybrid_without_query_text():
    dense = NumpyRetriever()
    sparse = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    chunks = _make_chunks()
    embeddings = _random_embeddings(4)
    hybrid.add(chunks, embeddings)

    results = hybrid.search(embeddings[2], top_k=3)
    assert len(results) > 0


def test_hybrid_rrf_scores_positive():
    dense = NumpyRetriever()
    sparse = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    chunks = _make_chunks()
    embeddings = _random_embeddings(4)
    hybrid.add(chunks, embeddings)

    results = hybrid.search(embeddings[0], top_k=4, query_text="FAISS similarity search vectors")
    for r in results:
        assert r.score > 0


def test_hybrid_weights():
    dense = NumpyRetriever()
    sparse = BM25Retriever()
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        dense_weight=0.8,
        sparse_weight=0.2,
    )

    chunks = _make_chunks()
    embeddings = _random_embeddings(4)
    hybrid.add(chunks, embeddings)

    results = hybrid.search(embeddings[0], top_k=4, query_text="Python")
    assert len(results) > 0
