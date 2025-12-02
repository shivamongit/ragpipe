"""Tests for chunker implementations."""

from ragpipe.core import Document
from ragpipe.chunkers.token import TokenChunker
from ragpipe.chunkers.recursive import RecursiveChunker


def test_token_chunker_basic():
    doc = Document(content="Hello world. " * 200)
    chunker = TokenChunker(chunk_size=64, overlap=8)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(c.doc_id == doc.doc_id for c in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_token_chunker_small_doc():
    doc = Document(content="Short text.")
    chunker = TokenChunker(chunk_size=512, overlap=64)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_token_chunker_empty_doc():
    doc = Document(content="")
    chunker = TokenChunker(chunk_size=512, overlap=64)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 0


def test_token_chunker_metadata():
    doc = Document(content="Hello world. " * 200, metadata={"source": "test.txt"})
    chunker = TokenChunker(chunk_size=64, overlap=8)
    chunks = chunker.chunk(doc)
    assert chunks[0].metadata["source"] == "test.txt"
    assert "token_count" in chunks[0].metadata


def test_recursive_chunker_basic():
    text = "Paragraph one about AI.\n\nParagraph two about ML.\n\nParagraph three about RAG."
    doc = Document(content=text)
    chunker = RecursiveChunker(chunk_size=512, overlap=0)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 1
    assert all(c.doc_id == doc.doc_id for c in chunks)


def test_recursive_chunker_respects_size():
    text = ("This is a sentence about topic A. " * 50 + "\n\n") * 10
    doc = Document(content=text)
    chunker = RecursiveChunker(chunk_size=64, overlap=0)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
