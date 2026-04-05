"""ChromaDB vector retriever — persistent local store, zero-config."""

from __future__ import annotations

from ragpipe.core import Chunk, RetrievalResult
from ragpipe.retrievers.base import BaseRetriever


class ChromaRetriever(BaseRetriever):
    """Vector retriever backed by ChromaDB.

    ChromaDB is an embedded vector database that persists to disk by default.
    Zero configuration needed — just point at a directory.

    Features:
    - Persistent storage across sessions
    - Metadata filtering on queries
    - Built-in embedding support (optional)
    - In-memory mode for testing

    Usage:
        retriever = ChromaRetriever(collection_name="docs", persist_dir="./chroma_db")
    """

    def __init__(
        self,
        collection_name: str = "ragpipe",
        persist_dir: str | None = None,
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Install chromadb: pip install 'ragpipe[chroma]'")

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                **{k: str(v) for k, v in chunk.metadata.items()},
            }
            for chunk in chunks
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        if self._collection.count() == 0:
            return []

        k = min(top_k, self._collection.count())
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        output = []
        for i in range(len(results["ids"][0])):
            doc_id = results["metadatas"][0][i].get("doc_id", "")
            chunk_index = int(results["metadatas"][0][i].get("chunk_index", 0))
            text = results["documents"][0][i]
            distance = results["distances"][0][i]
            score = 1.0 - distance

            meta = {k: v for k, v in results["metadatas"][0][i].items()
                    if k not in ("doc_id", "chunk_index")}

            chunk = Chunk(
                text=text,
                doc_id=doc_id,
                chunk_index=chunk_index,
                metadata=meta,
            )
            output.append(RetrievalResult(chunk=chunk, score=score))

        return output

    def delete(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document."""
        existing = self._collection.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
            return len(existing["ids"])
        return 0

    @property
    def count(self) -> int:
        return self._collection.count()
