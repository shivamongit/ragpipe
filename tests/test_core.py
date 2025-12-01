"""Tests for core data structures."""

from ragpipe.core import Document, Chunk, RetrievalResult


def test_document_auto_id():
    doc = Document(content="hello world")
    assert doc.doc_id
    assert len(doc.doc_id) == 12


def test_document_custom_id():
    doc = Document(content="hello", doc_id="custom-123")
    assert doc.doc_id == "custom-123"


def test_document_dedup():
    d1 = Document(content="same content")
    d2 = Document(content="same content")
    assert d1.doc_id == d2.doc_id


def test_chunk_id():
    chunk = Chunk(text="hello", doc_id="abc", chunk_index=3)
    assert chunk.id == "abc:3"


def test_retrieval_result_repr():
    chunk = Chunk(text="short text", doc_id="abc", chunk_index=0)
    result = RetrievalResult(chunk=chunk, score=0.95, rank=1)
    assert "0.95" in repr(result)
    assert "short text" in repr(result)
