"""Core data structures and pipeline orchestrator."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Document:
    """A source document with content and metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class Chunk:
    """A chunk of a document with position tracking."""

    text: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    @property
    def id(self) -> str:
        return f"{self.doc_id}:{self.chunk_index}"


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance score."""

    chunk: Chunk
    score: float
    rank: int = 0

    def __repr__(self) -> str:
        preview = self.chunk.text[:80] + "..." if len(self.chunk.text) > 80 else self.chunk.text
        return f"RetrievalResult(score={self.score:.4f}, rank={self.rank}, text='{preview}')"


@dataclass
class GenerationResult:
    """LLM generation output with metadata."""

    answer: str
    sources: list[RetrievalResult]
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Orchestrates the RAG pipeline: chunk → embed → index → retrieve → rerank → generate.

    Example:
        pipe = Pipeline(
            chunker=TokenChunker(chunk_size=512),
            embedder=OpenAIEmbedder(model="text-embedding-3-small"),
            retriever=FaissRetriever(dim=1536),
            reranker=CrossEncoderReranker(),
            generator=OpenAIGenerator(model="gpt-4o-mini"),
        )
        pipe.ingest([Document(content="...")])
        result = pipe.query("What is the main finding?")
    """

    def __init__(
        self,
        chunker,
        embedder,
        retriever,
        generator,
        reranker=None,
        top_k: int = 5,
        rerank_top_k: int = 3,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self._documents: list[Document] = []

    def ingest(self, documents: list[Document]) -> dict[str, int]:
        """Ingest documents: chunk, embed, and index."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"documents": 0, "chunks": 0}

        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed(texts)

        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        self.retriever.add(all_chunks, embeddings)
        self._documents.extend(documents)

        return {"documents": len(documents), "chunks": len(all_chunks)}

    def query(self, question: str, top_k: int | None = None) -> GenerationResult:
        """Full RAG query: embed → retrieve → rerank → generate."""
        t0 = time.perf_counter()
        k = top_k or self.top_k

        query_embedding = self.embedder.embed([question])[0]
        results = self.retriever.search(query_embedding, top_k=k)

        if self.reranker and results:
            results = self.reranker.rerank(question, results, top_k=self.rerank_top_k)

        for i, r in enumerate(results):
            r.rank = i + 1

        answer = self.generator.generate(question, results)

        latency = (time.perf_counter() - t0) * 1000
        return GenerationResult(
            answer=answer.answer,
            sources=results,
            model=answer.model,
            tokens_used=answer.tokens_used,
            latency_ms=round(latency, 2),
            metadata=answer.metadata,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve without generation — useful for debugging and evaluation."""
        k = top_k or self.top_k
        query_embedding = self.embedder.embed([question])[0]
        results = self.retriever.search(query_embedding, top_k=k)

        if self.reranker and results:
            results = self.reranker.rerank(question, results, top_k=self.rerank_top_k)

        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    async def aingest(self, documents: list[Document]) -> dict[str, int]:
        """Async ingest: chunk, embed, and index documents."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"documents": 0, "chunks": 0}

        texts = [c.text for c in all_chunks]
        embeddings = await self.embedder.aembed(texts)

        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        self.retriever.add(all_chunks, embeddings)
        self._documents.extend(documents)

        return {"documents": len(documents), "chunks": len(all_chunks)}

    async def aquery(self, question: str, top_k: int | None = None) -> GenerationResult:
        """Async full RAG query: embed → retrieve → rerank → generate."""
        t0 = time.perf_counter()
        k = top_k or self.top_k

        query_embedding = (await self.embedder.aembed([question]))[0]
        results = self.retriever.search(query_embedding, top_k=k)

        if self.reranker and results:
            results = await self.reranker.arerank(question, results, top_k=self.rerank_top_k)

        for i, r in enumerate(results):
            r.rank = i + 1

        answer = await self.generator.agenerate(question, results)

        latency = (time.perf_counter() - t0) * 1000
        return GenerationResult(
            answer=answer.answer,
            sources=results,
            model=answer.model,
            tokens_used=answer.tokens_used,
            latency_ms=round(latency, 2),
            metadata=answer.metadata,
        )

    async def aretrieve(self, question: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Async retrieve without generation."""
        k = top_k or self.top_k
        query_embedding = (await self.embedder.aembed([question]))[0]
        results = self.retriever.search(query_embedding, top_k=k)

        if self.reranker and results:
            results = await self.reranker.arerank(question, results, top_k=self.rerank_top_k)

        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    async def stream_query(self, question: str, top_k: int | None = None) -> AsyncIterator[str]:
        """Async streaming query: embed → retrieve → rerank → stream tokens."""
        k = top_k or self.top_k

        query_embedding = (await self.embedder.aembed([question]))[0]
        results = self.retriever.search(query_embedding, top_k=k)

        if self.reranker and results:
            results = await self.reranker.arerank(question, results, top_k=self.rerank_top_k)

        for i, r in enumerate(results):
            r.rank = i + 1

        async for token in self.generator.astream(question, results):
            yield token

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return self.retriever.count
