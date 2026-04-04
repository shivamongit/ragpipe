"""Tests for BM25 retriever."""

from ragpipe.core import Chunk
from ragpipe.retrievers.bm25_retriever import BM25Retriever


def _make_chunks():
    return [
        Chunk(text="Python is a programming language used for web development and data science", doc_id="doc1", chunk_index=0),
        Chunk(text="JavaScript is used for frontend web development and Node.js backend", doc_id="doc2", chunk_index=0),
        Chunk(text="FAISS is a library for efficient similarity search of dense vectors", doc_id="doc3", chunk_index=0),
        Chunk(text="BM25 is a ranking function used in information retrieval systems", doc_id="doc4", chunk_index=0),
        Chunk(text="RAG combines retrieval with language model generation for grounded answers", doc_id="doc5", chunk_index=0),
    ]


def test_bm25_add_and_search():
    retriever = BM25Retriever()
    chunks = _make_chunks()
    retriever.add(chunks)
    assert retriever.count == 5

    results = retriever.search_text("Python programming", top_k=3)
    assert len(results) > 0
    assert results[0].chunk.doc_id == "doc1"


def test_bm25_exact_keyword_match():
    retriever = BM25Retriever()
    chunks = _make_chunks()
    retriever.add(chunks)

    results = retriever.search_text("FAISS similarity search", top_k=3)
    assert results[0].chunk.doc_id == "doc3"


def test_bm25_empty():
    retriever = BM25Retriever()
    results = retriever.search_text("anything", top_k=5)
    assert results == []


def test_bm25_no_match():
    retriever = BM25Retriever()
    chunks = _make_chunks()
    retriever.add(chunks)

    results = retriever.search_text("quantum physics entanglement", top_k=3)
    assert len(results) == 0


def test_bm25_scores_decrease():
    retriever = BM25Retriever()
    chunks = _make_chunks()
    retriever.add(chunks)

    results = retriever.search_text("web development", top_k=5)
    assert len(results) >= 2
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
