"""Parent-Child chunker — small-to-big retrieval pattern.

Indexes small child chunks for precise embedding matching,
but returns the larger parent chunk for richer LLM context.

This is one of the highest-impact retrieval improvements:
small chunks embed well (focused semantics), but large chunks
generate better answers (more context).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragpipe.chunkers.base import BaseChunker

if TYPE_CHECKING:
    from ragpipe.core import Chunk, Document

class ParentChildChunker(BaseChunker):
    """Chunks documents at two levels: parent (large) and child (small).

    Child chunks are used for embedding/retrieval (precise matching).
    Each child stores its parent's text in metadata so the pipeline
    can return the full parent context to the generator.

    Chunk sizes are measured in *words* (whitespace-split tokens).
    This is fast and avoids external tokenizer dependencies.

    Usage:
        chunker = ParentChildChunker(
            parent_chunk_size=512,
            child_chunk_size=128,
            parent_overlap=64,
            child_overlap=16,
        )
        chunks = chunker.chunk(document)
        # Each chunk has metadata["parent_text"] with the larger context
    """

    def __init__(
        self,
        parent_chunk_size: int = 512,
        child_chunk_size: int = 128,
        parent_overlap: int = 64,
        child_overlap: int = 16,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap

    @staticmethod
    def _window_slices(words: list[str], size: int, overlap: int) -> list[list[str]]:
        """Split a word list into overlapping windows."""
        if not words:
            return []
        # Clamp overlap so the window always advances by at least 1
        overlap = min(overlap, size - 1) if size > 1 else 0
        windows: list[list[str]] = []
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            windows.append(words[start:end])
            if end >= len(words):
                break
            start = end - overlap
        return windows

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into child chunks with parent context in metadata."""
        from ragpipe.core import Chunk

        if not document.content.strip():
            return []

        words = document.content.split()
        if not words:
            return []

        # Step 1: Create parent windows (large word slices)
        parent_windows = self._window_slices(
            words, self.parent_chunk_size, self.parent_overlap
        )

        # Step 2: For each parent, create child windows (small word slices)
        all_children: list[Chunk] = []
        child_index = 0

        for parent_idx, parent_words in enumerate(parent_windows):
            parent_text = " ".join(parent_words)

            child_windows = self._window_slices(
                parent_words, self.child_chunk_size, self.child_overlap
            )

            for local_idx, child_words in enumerate(child_windows):
                child_text = " ".join(child_words)
                if not child_text:
                    continue

                metadata = {
                    **document.metadata,
                    "parent_text": parent_text,
                    "parent_index": parent_idx,
                    "child_index_in_parent": local_idx,
                    "chunker": "parent_child",
                }

                all_children.append(Chunk(
                    text=child_text,
                    doc_id=document.doc_id,
                    chunk_index=child_index,
                    metadata=metadata,
                ))
                child_index += 1

        return all_children
