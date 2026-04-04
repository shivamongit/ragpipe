"""Tests for contextual chunker."""

from ragpipe.core import Document
from ragpipe.chunkers.token import TokenChunker
from ragpipe.chunkers.contextual import ContextualChunker


def test_contextual_without_generator():
    """Without a context generator, should return base chunks unchanged."""
    base = TokenChunker(chunk_size=64, overlap=8)
    chunker = ContextualChunker(base_chunker=base)

    doc = Document(content="Hello world. " * 100)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    assert all(c.doc_id == doc.doc_id for c in chunks)


def test_contextual_with_generator():
    """With a context generator, chunks should be prepended with context."""
    base = TokenChunker(chunk_size=64, overlap=8)

    def fake_generator(prompt: str) -> str:
        return "This chunk discusses the main topic of the document."

    chunker = ContextualChunker(base_chunker=base, context_generator=fake_generator)

    doc = Document(content="Artificial intelligence is transforming industries. " * 50)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    assert chunks[0].text.startswith("This chunk discusses")
    assert chunks[0].metadata.get("has_context") is True
    assert "original_text" in chunks[0].metadata


def test_contextual_generator_failure():
    """If generator fails, chunk should fall back to original text."""
    base = TokenChunker(chunk_size=64, overlap=8)

    def failing_generator(prompt: str) -> str:
        raise RuntimeError("LLM unavailable")

    chunker = ContextualChunker(base_chunker=base, context_generator=failing_generator)

    doc = Document(content="Some content here. " * 50)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    # Should still have chunks, just without context prepended
    for c in chunks:
        assert c.text
