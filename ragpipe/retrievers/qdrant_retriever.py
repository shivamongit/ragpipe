"""Qdrant vector retriever — self-hosted or cloud, with metadata filtering."""

from __future__ import annotations

import uuid

from ragpipe.core import Chunk, RetrievalResult
from ragpipe.retrievers.base import BaseRetriever


class QdrantRetriever(BaseRetriever):
    """Vector retriever backed by Qdrant.

    Qdrant supports both self-hosted (Docker) and managed cloud deployment.
    Features metadata filtering, named vectors, and payload indexing.

    Usage:
        # In-memory (testing)
        retriever = QdrantRetriever(collection_name="docs", dim=768, location=":memory:")

        # Local persistent
        retriever = QdrantRetriever(collection_name="docs", dim=768, path="./qdrant_data")

        # Cloud / self-hosted
        retriever = QdrantRetriever(collection_name="docs", dim=768, url="http://localhost:6333")
    """

    def __init__(
        self,
        collection_name: str = "ragpipe",
        dim: int = 768,
        location: str | None = None,
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("Install qdrant-client: pip install 'ragpipe[qdrant]'")

        self._dim = dim

        if location == ":memory:":
            self._client = QdrantClient(location=":memory:")
        elif path:
            self._client = QdrantClient(path=path)
        elif url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(location=":memory:")

        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        self._collection_name = collection_name

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for chunk, emb in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_id": chunk.id,
                        **{k: str(v) for k, v in chunk.metadata.items()},
                    },
                )
            )

        self._client.upsert(collection_name=self._collection_name, points=points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_conditions: dict | None = None,
    ) -> list[RetrievalResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        output = []
        for hit in results:
            payload = hit.payload or {}
            chunk = Chunk(
                text=payload.get("text", ""),
                doc_id=payload.get("doc_id", ""),
                chunk_index=int(payload.get("chunk_index", 0)),
                metadata={k: v for k, v in payload.items()
                          if k not in ("text", "doc_id", "chunk_index", "chunk_id")},
            )
            output.append(RetrievalResult(chunk=chunk, score=hit.score))

        return output

    def delete(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        before = self.count
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        return before - self.count

    @property
    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0
