"""Tests for ParentChildChunker."""

from ragpipe.core import Document
from ragpipe.chunkers.parent_child import ParentChildChunker


def test_parent_child_basic():
    text = " ".join([f"word{i}" for i in range(200)])
    doc = Document(content=text)
    chunker = ParentChildChunker(
        parent_chunk_size=100, child_chunk_size=30, parent_overlap=10, child_overlap=5
    )
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    # Every child should have parent_text in metadata
    for c in chunks:
        assert "parent_text" in c.metadata
        assert len(c.metadata["parent_text"]) >= len(c.text)


def test_parent_child_metadata():
    text = " ".join([f"sentence{i}" for i in range(30)])
    doc = Document(content=text, metadata={"source": "test.txt"})
    chunker = ParentChildChunker(parent_chunk_size=20, child_chunk_size=8)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    assert chunks[0].metadata["source"] == "test.txt"
    assert chunks[0].metadata["chunker"] == "parent_child"
    assert "parent_index" in chunks[0].metadata


def test_parent_child_empty():
    doc = Document(content="")
    chunker = ParentChildChunker()
    chunks = chunker.chunk(doc)
    assert chunks == []


def test_parent_child_small_doc():
    doc = Document(content="Hello world.")
    chunker = ParentChildChunker(parent_chunk_size=1024, child_chunk_size=256)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "Hello world."


def test_parent_child_chunk_ids_unique():
    text = " ".join([f"token{i}" for i in range(50)])
    doc = Document(content=text)
    chunker = ParentChildChunker(parent_chunk_size=30, child_chunk_size=10)
    chunks = chunker.chunk(doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "All chunk IDs should be unique"
