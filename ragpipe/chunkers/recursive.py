"""Recursive character-based chunker with hierarchical separators."""

from __future__ import annotations

from ragpipe.core import Chunk, Document
from ragpipe.chunkers.base import BaseChunker

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Split documents recursively by trying separators in order.

    Tries the largest separator first (paragraph), then falls back to
    sentence, word, and character-level splitting. This preserves
    semantic coherence better than fixed-window approaches.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
        encoding: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self._enc = self._load_encoder(encoding)

    @staticmethod
    def _load_encoder(encoding: str):
        from rs_bpe.bpe import openai
        encoders = {
            "cl100k_base": openai.cl100k_base,
            "o200k_base": openai.o200k_base,
        }
        factory = encoders.get(encoding)
        if factory is None:
            raise ValueError(f"Unknown encoding {encoding!r}. Available: {list(encoders)}")
        return factory()

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []

        if self._token_len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if sep == "":
            tokens = self._enc.encode(text)
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                decoded = self._enc.decode(tokens[start:end]).strip()
                if decoded:
                    chunks.append(decoded)
                start = end - self.overlap if end < len(tokens) else end
            return chunks

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()

            if self._token_len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    if self._token_len(current) <= self.chunk_size:
                        chunks.append(current.strip())
                    else:
                        chunks.extend(self._split_text(current, remaining_seps))

                if self._token_len(part) <= self.chunk_size:
                    current = part.strip()
                else:
                    chunks.extend(self._split_text(part, remaining_seps))
                    current = ""

        if current.strip():
            if self._token_len(current) <= self.chunk_size:
                chunks.append(current.strip())
            else:
                chunks.extend(self._split_text(current, remaining_seps))

        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        texts = self._split_text(document.content, self.separators)
        chunks = []
        for idx, text in enumerate(texts):
            if text:
                chunks.append(
                    Chunk(
                        text=text,
                        doc_id=document.doc_id,
                        chunk_index=idx,
                        metadata={
                            **document.metadata,
                            "chunker": "recursive",
                            "token_count": self._token_len(text),
                        },
                    )
                )
        return chunks
