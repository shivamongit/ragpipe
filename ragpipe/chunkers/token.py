"""Token-based chunker using tiktoken."""

from __future__ import annotations

import tiktoken

from ragpipe.core import Chunk, Document
from ragpipe.chunkers.base import BaseChunker


class TokenChunker(BaseChunker):
    """Split documents by token count with configurable overlap.

    Uses tiktoken's cl100k_base encoding (same as GPT-4, text-embedding-3).
    Overlap ensures context continuity across chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        encoding: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(encoding)

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = self._enc.encode(document.content)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            text = self._enc.decode(tokens[start:end]).strip()

            if text:
                chunks.append(
                    Chunk(
                        text=text,
                        doc_id=document.doc_id,
                        chunk_index=idx,
                        metadata={
                            **document.metadata,
                            "token_start": start,
                            "token_end": end,
                            "token_count": end - start,
                        },
                    )
                )
                idx += 1

            if end >= len(tokens):
                break
            start = end - self.overlap

        return chunks
