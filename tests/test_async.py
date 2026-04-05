"""Tests for async Pipeline methods (aingest, aquery, aretrieve, stream_query)."""

import asyncio
import numpy as np
import pytest

from ragpipe.core import Document, Chunk, RetrievalResult, Pipeline
from ragpipe.chunkers.token import TokenChunker
from ragpipe.embedders.base import BaseEmbedder
from ragpipe.retrievers.numpy_retriever import NumpyRetriever
from ragpipe.generators.base import BaseGenerator, GenerationOutput


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder for testing."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(texts), 32)).tolist()

    @property
    def dim(self) -> int:
        return 32


class MockGenerator(BaseGenerator):
    """Deterministic generator for testing."""

    def generate(self, question: str, context: list[RetrievalResult]) -> GenerationOutput:
        return GenerationOutput(
            answer=f"Answer to: {question}",
            model="mock",
            tokens_used=10,
        )


def _make_pipeline():
    return Pipeline(
        chunker=TokenChunker(chunk_size=64, overlap=8),
        embedder=MockEmbedder(),
        retriever=NumpyRetriever(),
        generator=MockGenerator(),
        top_k=3,
    )


@pytest.mark.asyncio
async def test_aingest():
    pipe = _make_pipeline()
    docs = [Document(content="Hello world. " * 50)]
    stats = await pipe.aingest(docs)
    assert stats["documents"] == 1
    assert stats["chunks"] > 0
    assert pipe.chunk_count > 0


@pytest.mark.asyncio
async def test_aquery():
    pipe = _make_pipeline()
    docs = [Document(content="Python is a programming language. " * 50)]
    await pipe.aingest(docs)

    result = await pipe.aquery("What is Python?")
    assert result.answer == "Answer to: What is Python?"
    assert result.model == "mock"
    assert result.latency_ms > 0


@pytest.mark.asyncio
async def test_aretrieve():
    pipe = _make_pipeline()
    docs = [Document(content="FAISS enables fast similarity search. " * 50)]
    await pipe.aingest(docs)

    results = await pipe.aretrieve("What is FAISS?")
    assert len(results) > 0
    assert all(r.rank > 0 for r in results)


@pytest.mark.asyncio
async def test_stream_query():
    pipe = _make_pipeline()
    docs = [Document(content="Streaming test content. " * 50)]
    await pipe.aingest(docs)

    tokens = []
    async for token in pipe.stream_query("What is streaming?"):
        tokens.append(token)

    assert len(tokens) > 0
    full_answer = "".join(tokens)
    assert "What is streaming?" in full_answer


@pytest.mark.asyncio
async def test_aingest_empty():
    pipe = _make_pipeline()
    stats = await pipe.aingest([Document(content="")])
    assert stats["chunks"] == 0


@pytest.mark.asyncio
async def test_aembed_default():
    """Test that BaseEmbedder.aembed default uses asyncio.to_thread."""
    embedder = MockEmbedder()
    result = await embedder.aembed(["hello", "world"])
    assert len(result) == 2
    assert len(result[0]) == 32
