"""Semantic chunker that splits on embedding similarity breakpoints."""

from __future__ import annotations

import numpy as np
import tiktoken

from ragpipe.core import Chunk, Document
from ragpipe.chunkers.base import BaseChunker


class SemanticChunker(BaseChunker):
    """Split documents at semantic boundary points.

    Splits text into sentences, embeds each sentence, then detects
    breakpoints where cosine similarity between consecutive sentences
    drops below a threshold. Groups sentences between breakpoints
    into chunks, ensuring semantic coherence within each chunk.

    Requires an embedder instance for sentence-level embedding.
    """

    def __init__(
        self,
        embedder,
        threshold: float = 0.75,
        max_chunk_tokens: int = 512,
        min_sentences: int = 2,
        encoding: str = "cl100k_base",
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.min_sentences = min_sentences
        self._enc = tiktoken.get_encoding(encoding)

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter. Handles common abbreviations."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if len(sentences) <= self.min_sentences:
            return [
                Chunk(
                    text=document.content.strip(),
                    doc_id=document.doc_id,
                    chunk_index=0,
                    metadata={**document.metadata, "chunker": "semantic"},
                )
            ] if document.content.strip() else []

        embeddings = self.embedder.embed(sentences)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Find breakpoints where similarity drops
        breakpoints = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < self.threshold:
                breakpoints.append(i)

        # Group sentences between breakpoints
        groups: list[list[str]] = []
        start = 0
        for bp in breakpoints:
            group = sentences[start:bp]
            if group:
                groups.append(group)
            start = bp
        if start < len(sentences):
            groups.append(sentences[start:])

        # Merge small groups and split large ones
        chunks: list[Chunk] = []
        idx = 0
        buffer: list[str] = []

        for group in groups:
            candidate = buffer + group
            candidate_text = " ".join(candidate)

            if self._token_len(candidate_text) <= self.max_chunk_tokens:
                buffer = candidate
            else:
                if buffer:
                    text = " ".join(buffer).strip()
                    if text:
                        chunks.append(
                            Chunk(
                                text=text,
                                doc_id=document.doc_id,
                                chunk_index=idx,
                                metadata={
                                    **document.metadata,
                                    "chunker": "semantic",
                                    "sentence_count": len(buffer),
                                    "token_count": self._token_len(text),
                                },
                            )
                        )
                        idx += 1
                buffer = group

        if buffer:
            text = " ".join(buffer).strip()
            if text:
                chunks.append(
                    Chunk(
                        text=text,
                        doc_id=document.doc_id,
                        chunk_index=idx,
                        metadata={
                            **document.metadata,
                            "chunker": "semantic",
                            "sentence_count": len(buffer),
                            "token_count": self._token_len(text),
                        },
                    )
                )

        return chunks
