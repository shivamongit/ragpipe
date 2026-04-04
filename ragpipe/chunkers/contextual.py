"""Contextual chunker — Anthropic's contextual retrieval approach.

Prepends each chunk with LLM-generated context about where it fits
in the original document. This dramatically improves retrieval accuracy
(Anthropic reported 49% fewer retrieval failures).
"""

from __future__ import annotations

from ragpipe.core import Chunk, Document
from ragpipe.chunkers.base import BaseChunker

CONTEXT_PROMPT = """<document>
{doc_preview}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the context, nothing else."""


class ContextualChunker(BaseChunker):
    """Two-stage chunker: first chunk normally, then prepend document context.

    Uses a base chunker to split the document, then calls an LLM to generate
    a 2-3 sentence context prefix for each chunk explaining where it fits
    in the original document. The context is prepended to improve retrieval.

    Parameters:
        base_chunker: Any BaseChunker to perform initial splitting
        context_generator: Callable that takes a prompt string and returns context string
        doc_preview_chars: How many characters of the document to show to the LLM (default 3000)
    """

    def __init__(
        self,
        base_chunker: BaseChunker,
        context_generator=None,
        doc_preview_chars: int = 3000,
    ):
        self.base_chunker = base_chunker
        self._generate_context = context_generator
        self.doc_preview_chars = doc_preview_chars

    def chunk(self, document: Document) -> list[Chunk]:
        base_chunks = self.base_chunker.chunk(document)

        if not self._generate_context:
            return base_chunks

        doc_preview = document.content[: self.doc_preview_chars]
        contextualized: list[Chunk] = []

        for chunk in base_chunks:
            prompt = CONTEXT_PROMPT.format(
                doc_preview=doc_preview,
                chunk_text=chunk.text,
            )

            try:
                context = self._generate_context(prompt)
                enriched_text = f"{context.strip()}\n\n{chunk.text}"
            except Exception:
                enriched_text = chunk.text

            contextualized.append(
                Chunk(
                    text=enriched_text,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    metadata={
                        **chunk.metadata,
                        "chunker": "contextual",
                        "has_context": True,
                        "original_text": chunk.text,
                    },
                )
            )

        return contextualized
