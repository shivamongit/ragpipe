"""BM25 sparse keyword retriever — no vector embeddings needed."""

from __future__ import annotations

import math
import re
from collections import Counter

from ragpipe.core import Chunk, RetrievalResult
from ragpipe.retrievers.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Okapi BM25 keyword retriever using pure Python.

    BM25 captures exact keyword matches that dense embeddings miss.
    Pair with a dense retriever via HybridRetriever for best results.

    Parameters:
        k1: Term frequency saturation parameter (default 1.5)
        b:  Length normalization parameter (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[dict] = []
        self._doc_freqs: Counter = Counter()
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._tf_cache: list[Counter] = []
        self._n_docs: int = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        return re.findall(r"\b\w+\b", text.lower())

    def add(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None:
        """Index chunks for BM25 search. Embeddings are ignored."""
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tf = Counter(tokens)
            self._tf_cache.append(tf)
            self._doc_lens.append(len(tokens))

            for term in set(tokens):
                self._doc_freqs[term] += 1

            self._chunks.append({
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            })

        self._n_docs = len(self._chunks)
        self._avg_dl = sum(self._doc_lens) / self._n_docs if self._n_docs else 0.0

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        tf = self._tf_cache[doc_idx]
        dl = self._doc_lens[doc_idx]
        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue
            n_t = self._doc_freqs.get(term, 0)
            idf = math.log((self._n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)
            term_freq = tf[term]
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * (numerator / denominator)

        return score

    def search(self, query_embedding: list[float] | str, top_k: int = 5) -> list[RetrievalResult]:
        """Search by keyword. Pass query text as string, or embedding is ignored."""
        if isinstance(query_embedding, str):
            query_text = query_embedding
        else:
            return []

        if not self._chunks:
            return []

        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        scores = []
        for i in range(self._n_docs):
            s = self._bm25_score(query_tokens, i)
            if s > 0:
                scores.append((i, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        results = []
        for idx, score in top:
            meta = self._chunks[idx]
            chunk = Chunk(
                text=meta["text"],
                doc_id=meta["doc_id"],
                chunk_index=meta["chunk_index"],
                metadata=meta["metadata"],
            )
            results.append(RetrievalResult(chunk=chunk, score=score))

        return results

    def search_text(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Convenience method: search by raw text query."""
        return self.search(query, top_k=top_k)

    @property
    def count(self) -> int:
        return len(self._chunks)
