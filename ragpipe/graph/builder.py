"""Build a knowledge graph from documents by extracting entities and relationships."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ragpipe.graph.entities import Entity, EntityExtractor, KnowledgeGraph, Relationship


@dataclass
class GraphBuildResult:
    """Result of building or extending a knowledge graph."""

    graph: KnowledgeGraph
    documents_processed: int
    entities_extracted: int
    relationships_extracted: int
    build_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"GraphBuildResult(docs={self.documents_processed}, "
            f"entities={self.entities_extracted}, rels={self.relationships_extracted}, "
            f"time={self.build_time_ms:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Similarity helpers for entity deduplication
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(name.lower().split())


def _char_bigrams(text: str) -> set[str]:
    """Return the set of character bigrams for *text*."""
    t = _normalize(text)
    return {t[i: i + 2] for i in range(max(len(t) - 1, 0))}


def _bigram_similarity(a: str, b: str) -> float:
    """Sørensen–Dice coefficient over character bigrams."""
    ba, bb = _char_bigrams(a), _char_bigrams(b)
    if not ba or not bb:
        return 0.0
    return 2 * len(ba & bb) / (len(ba) + len(bb))


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Build a knowledge graph from documents by extracting entities and relationships.

    Accepts plain strings or any object with a ``.content`` attribute (e.g.
    :class:`ragpipe.core.Document`).  After extraction, near-duplicate entities
    (measured by character-bigram similarity) are merged when their similarity
    exceeds *merge_threshold*.

    Example::

        builder = GraphBuilder()
        result = builder.build(["Alice works at Acme Corp.", "Bob joined Acme Inc."])
        print(result.graph.stats())
    """

    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        merge_threshold: float = 0.85,
    ) -> None:
        self._extractor = extractor or EntityExtractor()
        self.merge_threshold = merge_threshold

    # -- public API ---------------------------------------------------------

    def build(
        self,
        documents: list[Any],
        chunks: Optional[list[Any]] = None,
    ) -> GraphBuildResult:
        """Build a knowledge graph from *documents* (and optional *chunks*).

        Each element may be a string or an object with a ``.content`` attribute
        and an optional ``.doc_id`` attribute.
        """
        start = time.perf_counter()
        graph = KnowledgeGraph()
        sources = list(documents)
        if chunks:
            sources.extend(chunks)

        for src in sources:
            text, doc_id = self._unpack(src)
            if not text:
                continue
            entities, rels = self._extractor.extract(text, doc_id)
            for entity in entities:
                graph.add_entity(entity)
            for rel in rels:
                graph.add_relationship(rel)

        graph = self._merge_similar_entities(graph)
        elapsed = (time.perf_counter() - start) * 1000
        return GraphBuildResult(
            graph=graph,
            documents_processed=len(sources),
            entities_extracted=graph.entity_count,
            relationships_extracted=graph.relationship_count,
            build_time_ms=elapsed,
        )

    async def abuild(
        self,
        documents: list[Any],
        chunks: Optional[list[Any]] = None,
    ) -> GraphBuildResult:
        """Async build. Override for native async; default wraps sync in a thread."""
        return await asyncio.to_thread(self.build, documents, chunks)

    def add_documents(
        self,
        graph: KnowledgeGraph,
        documents: list[Any],
    ) -> GraphBuildResult:
        """Extract entities/relationships from *documents* and merge into an existing *graph*."""
        start = time.perf_counter()
        count_before_e = graph.entity_count
        count_before_r = graph.relationship_count

        for src in documents:
            text, doc_id = self._unpack(src)
            if not text:
                continue
            entities, rels = self._extractor.extract(text, doc_id)
            for entity in entities:
                graph.add_entity(entity)
            for rel in rels:
                graph.add_relationship(rel)

        graph = self._merge_similar_entities(graph)
        elapsed = (time.perf_counter() - start) * 1000
        return GraphBuildResult(
            graph=graph,
            documents_processed=len(documents),
            entities_extracted=graph.entity_count - count_before_e,
            relationships_extracted=graph.relationship_count - count_before_r,
            build_time_ms=elapsed,
        )

    # -- internals ----------------------------------------------------------

    def _merge_similar_entities(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        """Merge entities whose names are similar above *merge_threshold*.

        The entity with the longer description is kept as the canonical entry;
        relationships referencing the duplicate are re-pointed.
        """
        ids = list(graph.entities.keys())
        merge_map: dict[str, str] = {}  # old_id -> canonical_id

        for i, id_a in enumerate(ids):
            if id_a in merge_map:
                continue
            entity_a = graph.entities[id_a]
            for id_b in ids[i + 1:]:
                if id_b in merge_map:
                    continue
                entity_b = graph.entities[id_b]
                if entity_a.entity_type != entity_b.entity_type:
                    continue
                sim = _bigram_similarity(entity_a.name, entity_b.name)
                if sim >= self.merge_threshold:
                    # Keep the entity with more description as canonical
                    if len(entity_b.description) > len(entity_a.description):
                        merge_map[id_a] = id_b
                        break
                    else:
                        merge_map[id_b] = id_a

        if not merge_map:
            return graph

        # Rebuild graph with merged entities
        merged = KnowledgeGraph()
        for eid, entity in graph.entities.items():
            if eid not in merge_map:
                merged.add_entity(entity)

        for rel in graph.relationships:
            src = merge_map.get(rel.source, rel.source)
            tgt = merge_map.get(rel.target, rel.target)
            if src == tgt:
                continue
            merged.add_relationship(Relationship(
                source=src,
                target=tgt,
                relation_type=rel.relation_type,
                weight=rel.weight,
                properties=rel.properties,
                source_doc_id=rel.source_doc_id,
            ))
        return merged

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _unpack(source: Any) -> tuple[str, str]:
        """Return ``(text, doc_id)`` from a string or document-like object."""
        if isinstance(source, str):
            return source, ""
        text = getattr(source, "content", None) or getattr(source, "text", None) or ""
        doc_id = getattr(source, "doc_id", "") or ""
        return str(text), str(doc_id)
