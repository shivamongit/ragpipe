"""Tests for Agentic RAG query router."""

import pytest
import json

from ragpipe.agents.router import (
    QueryRouter, RouteDecision, RouteType, _parse_route_response, ROUTER_PROMPT,
)
from ragpipe.core import (
    Chunk, Document, GenerationResult, Pipeline, RetrievalResult,
)
from ragpipe.chunkers import TokenChunker
from ragpipe.embedders.base import BaseEmbedder
from ragpipe.generators.base import BaseGenerator, GenerationOutput
from ragpipe.retrievers import NumpyRetriever


# --- Mocks ---

class _MockEmbedder(BaseEmbedder):
    def embed(self, texts):
        return [[0.1] * 8 for _ in texts]

    @property
    def dim(self):
        return 8


class _MockGenerator(BaseGenerator):
    def generate(self, question, context):
        return GenerationOutput(answer=f"answer:{question}", model="mock")


# --- Tests ---

def test_parse_route_single():
    resp = '{"route": "single", "sub_questions": [], "reasoning": "simple"}'
    d = _parse_route_response(resp)
    assert d.route == RouteType.SINGLE
    assert d.sub_questions == []


def test_parse_route_multi_step():
    resp = json.dumps({
        "route": "multi_step",
        "sub_questions": ["q1?", "q2?"],
        "reasoning": "complex query",
    })
    d = _parse_route_response(resp)
    assert d.route == RouteType.MULTI_STEP
    assert d.sub_questions == ["q1?", "q2?"]


def test_parse_route_fallback():
    d = _parse_route_response("garbage output no json")
    assert d.route == RouteType.SINGLE


def test_parse_route_direct():
    resp = '{"route": "direct", "sub_questions": [], "reasoning": "general knowledge"}'
    d = _parse_route_response(resp)
    assert d.route == RouteType.DIRECT


def _make_pipeline():
    return Pipeline(
        chunker=TokenChunker(chunk_size=32, overlap=4),
        embedder=_MockEmbedder(),
        retriever=NumpyRetriever(),
        generator=_MockGenerator(),
    )


def test_router_no_classify_fn():
    pipe = _make_pipeline()
    router = QueryRouter(pipeline=pipe)
    decision = router.classify("What is X?")
    assert decision.route == RouteType.SINGLE


def test_router_single_route():
    pipe = _make_pipeline()
    pipe.ingest([Document(content="FAISS is a vector search library by Meta.")])

    def classify_fn(prompt):
        return '{"route": "single", "sub_questions": [], "reasoning": "simple"}'

    router = QueryRouter(pipeline=pipe, classify_fn=classify_fn)
    result = router.query("What is FAISS?")
    assert "answer:" in result.answer


def test_router_direct_route():
    pipe = _make_pipeline()

    def classify_fn(prompt):
        return '{"route": "direct", "sub_questions": [], "reasoning": "general knowledge"}'

    router = QueryRouter(pipeline=pipe, classify_fn=classify_fn)
    result = router.query("What is 2+2?")
    assert result.metadata.get("route") == "direct"
    assert result.sources == []


def test_router_multi_step():
    pipe = _make_pipeline()
    pipe.ingest([Document(content="FAISS is fast. ChromaDB is persistent. Qdrant is scalable.")])

    def classify_fn(prompt):
        return json.dumps({
            "route": "multi_step",
            "sub_questions": ["What is FAISS?", "What is ChromaDB?"],
            "reasoning": "comparison",
        })

    router = QueryRouter(pipeline=pipe, classify_fn=classify_fn)
    result = router.query("Compare FAISS and ChromaDB")
    assert result.metadata.get("route") == "multi_step"
    assert "sub_questions" in result.metadata


@pytest.mark.asyncio
async def test_router_aquery_single():
    pipe = _make_pipeline()
    pipe.ingest([Document(content="Test document content.")])

    def classify_fn(prompt):
        return '{"route": "single", "sub_questions": [], "reasoning": "simple"}'

    router = QueryRouter(pipeline=pipe, classify_fn=classify_fn)
    result = await router.aquery("What is this?")
    assert "answer:" in result.answer


@pytest.mark.asyncio
async def test_router_aquery_multi_step_parallel():
    pipe = _make_pipeline()
    pipe.ingest([Document(content="Alpha beta gamma delta epsilon zeta eta theta.")])

    async def aclassify_fn(prompt):
        return json.dumps({
            "route": "multi_step",
            "sub_questions": ["What is alpha?", "What is beta?"],
            "reasoning": "multiple topics",
        })

    router = QueryRouter(pipeline=pipe, aclassify_fn=aclassify_fn)
    result = await router.aquery("Compare alpha and beta")
    assert result.metadata["route"] == "multi_step"
