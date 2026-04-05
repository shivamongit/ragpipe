"""Tests for conversation memory."""

import pytest

from ragpipe.memory.conversation import ConversationMemory, Message
from ragpipe.core import Document, GenerationResult, Pipeline
from ragpipe.chunkers import TokenChunker
from ragpipe.embedders.base import BaseEmbedder
from ragpipe.generators.base import BaseGenerator, GenerationOutput
from ragpipe.retrievers import NumpyRetriever


class _MockEmbedder(BaseEmbedder):
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    @property
    def dim(self):
        return 8


class _MockGenerator(BaseGenerator):
    def generate(self, question, context):
        return GenerationOutput(answer=f"answer:{question}", model="mock")


def _make_pipeline():
    pipe = Pipeline(
        chunker=TokenChunker(chunk_size=32, overlap=4),
        embedder=_MockEmbedder(),
        retriever=NumpyRetriever(),
        generator=_MockGenerator(),
    )
    pipe.ingest([Document(content="FAISS is a vector search library developed by Meta.")])
    return pipe


def test_memory_add_message():
    mem = ConversationMemory()
    mem.add_message("user", "Hello")
    mem.add_message("assistant", "Hi there")
    assert len(mem.history) == 2
    assert mem.turn_count == 1


def test_memory_format_history():
    mem = ConversationMemory()
    mem.add_message("user", "What is FAISS?")
    mem.add_message("assistant", "FAISS is a library by Meta.")
    text = mem.format_history()
    assert "User: What is FAISS?" in text
    assert "Assistant: FAISS is a library by Meta." in text


def test_memory_format_history_empty():
    mem = ConversationMemory()
    text = mem.format_history()
    assert "no prior conversation" in text


def test_memory_contextualize_no_history():
    mem = ConversationMemory(contextualize_fn=lambda p: "rewritten")
    result = mem.contextualize("What about it?")
    # No history → return unchanged
    assert result == "What about it?"


def test_memory_contextualize_with_history():
    def mock_contextualize(prompt):
        return "What is the performance of FAISS?"

    mem = ConversationMemory(contextualize_fn=mock_contextualize)
    mem.add_message("user", "What is FAISS?")
    mem.add_message("assistant", "FAISS is a library by Meta.")

    result = mem.contextualize("What about its performance?")
    assert result == "What is the performance of FAISS?"


def test_memory_contextualize_no_fn():
    mem = ConversationMemory()
    mem.add_message("user", "prior message")
    result = mem.contextualize("follow-up?")
    assert result == "follow-up?"


def test_memory_query():
    pipe = _make_pipeline()
    mem = ConversationMemory()

    result = mem.query(pipe, "What is FAISS?")
    assert "answer:" in result.answer
    assert len(mem.history) == 2  # user + assistant
    assert mem.history[0].role == "user"
    assert mem.history[1].role == "assistant"


def test_memory_multi_turn():
    pipe = _make_pipeline()
    mem = ConversationMemory()

    mem.query(pipe, "What is FAISS?")
    mem.query(pipe, "Tell me more")
    assert mem.turn_count == 2
    assert len(mem.history) == 4


def test_memory_max_history():
    mem = ConversationMemory(max_history=4)
    for i in range(10):
        mem.add_message("user", f"msg{i}")
    assert len(mem.history) == 4


def test_memory_context_window():
    mem = ConversationMemory(context_window=3)
    for i in range(10):
        mem.add_message("user", f"msg{i}")
    window = mem.get_context_window()
    assert len(window) == 3
    assert window[0].content == "msg7"


def test_memory_clear():
    mem = ConversationMemory()
    mem.add_message("user", "hello")
    mem.clear()
    assert len(mem.history) == 0
    assert mem.turn_count == 0


def test_memory_query_metadata():
    pipe = _make_pipeline()

    def mock_ctx(prompt):
        return "Standalone question about FAISS performance"

    mem = ConversationMemory(contextualize_fn=mock_ctx)
    mem.add_message("user", "What is FAISS?")
    mem.add_message("assistant", "A library.")

    result = mem.query(pipe, "What about performance?")
    assert result.metadata["original_question"] == "What about performance?"
    assert "Standalone" in result.metadata["standalone_question"]


@pytest.mark.asyncio
async def test_memory_aquery():
    pipe = _make_pipeline()
    mem = ConversationMemory()

    result = await mem.aquery(pipe, "What is FAISS?")
    assert "answer:" in result.answer
    assert mem.turn_count == 1


@pytest.mark.asyncio
async def test_memory_acontextualize():
    async def amock_ctx(prompt):
        return "Standalone question"

    mem = ConversationMemory(acontextualize_fn=amock_ctx)
    mem.add_message("user", "prior")
    result = await mem.acontextualize("follow-up?")
    assert result == "Standalone question"
