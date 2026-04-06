"""Knowledge Graph — entity extraction, graph construction, and graph+vector retrieval fusion.

Builds a knowledge graph from ingested documents using LLM-based entity/relation
extraction, stores triples in a lightweight in-memory graph, and provides graph-based
retrieval that fuses with vector search for multi-hop reasoning.

Usage:
    from ragpipe.graph import KnowledgeGraph

    kg = KnowledgeGraph(extract_fn=my_llm)
    kg.add_document("Paris is the capital of France. France is in Europe.")
    results = kg.search("What continent is Paris in?", max_hops=2)

    # Fusion with vector retrieval:
    graph_results = kg.search("relationship between X and Y")
    vector_results = retriever.search(embedding)
    fused = kg.fuse(graph_results, vector_results)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Entity:
    """A named entity in the knowledge graph."""
    name: str
    entity_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    mentions: int = 1

    @property
    def id(self) -> str:
        return self.name.lower().strip()


@dataclass
class Relation:
    """A typed relationship between two entities."""
    relation_type: str
    weight: float = 1.0
    source_doc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Triple:
    """A subject-predicate-object triple."""
    subject: str
    predicate: str
    object: str
    weight: float = 1.0
    source_doc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "weight": self.weight,
            "source_doc": self.source_doc,
        }


@dataclass
class GraphSearchResult:
    """Result from a graph search operation."""
    entity: str
    related_entities: list[dict[str, Any]]
    paths: list[list[str]]
    triples: list[Triple]
    score: float = 0.0
    hops: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "related_entities": self.related_entities,
            "paths": self.paths,
            "triples": [t.to_dict() for t in self.triples],
            "score": self.score,
            "hops": self.hops,
        }


# Prompt for LLM-based entity/relation extraction
EXTRACTION_PROMPT = """Extract all entities and relationships from the following text.

Return a JSON array of triples in this exact format:
[
  {{"subject": "Entity A", "predicate": "relationship", "object": "Entity B"}},
  ...
]

Rules:
1. Entities should be proper nouns, concepts, or named things
2. Predicates should be clear relationship types (e.g., "is_capital_of", "located_in", "founded_by")
3. Normalize entity names (consistent casing, no abbreviations)
4. Extract ALL relationships, including implicit ones

Text:
{text}

JSON triples:"""


def _parse_triples(raw: str) -> list[Triple]:
    """Parse LLM output into Triple objects."""
    # Try JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [
                Triple(
                    subject=t.get("subject", ""),
                    predicate=t.get("predicate", ""),
                    object=t.get("object", ""),
                )
                for t in data
                if t.get("subject") and t.get("object")
            ]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON array from text
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return [
                Triple(
                    subject=t.get("subject", ""),
                    predicate=t.get("predicate", ""),
                    object=t.get("object", ""),
                )
                for t in data
                if t.get("subject") and t.get("object")
            ]
        except (json.JSONDecodeError, TypeError):
            pass

    return []


def _heuristic_extract(text: str) -> list[Triple]:
    """Simple heuristic extraction when no LLM is available.

    Extracts "Subject verb Object" patterns from sentences.
    """
    triples = []
    sentences = re.split(r'[.!?]+', text)

    # Pattern: "X is/are/was Y"
    is_pattern = re.compile(
        r'([A-Z][a-zA-Z\s]+?)\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?(.+)',
        re.IGNORECASE,
    )
    # Pattern: "X verb Y" for specific relationship verbs
    rel_pattern = re.compile(
        r'([A-Z][a-zA-Z\s]+?)\s+'
        r'(?:located\s+in|part\s+of|belongs?\s+to|founded\s+by|created\s+by|'
        r'capital\s+of|contains?|includes?|connects?\s+to|borders?)\s+'
        r'(.+)',
        re.IGNORECASE,
    )

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Try relationship pattern first
        m = rel_pattern.search(sentence)
        if m:
            subj = m.group(1).strip()
            obj_text = m.group(2).strip().rstrip('.')
            pred_match = re.search(
                r'(located\s+in|part\s+of|belongs?\s+to|founded\s+by|created\s+by|'
                r'capital\s+of|contains?|includes?|connects?\s+to|borders?)',
                sentence, re.IGNORECASE,
            )
            pred = pred_match.group(1).replace(' ', '_') if pred_match else "related_to"
            triples.append(Triple(subject=subj, predicate=pred, object=obj_text))
            continue

        # Try is/are pattern
        m = is_pattern.search(sentence)
        if m:
            subj = m.group(1).strip()
            obj_text = m.group(2).strip().rstrip('.')
            if len(subj) > 1 and len(obj_text) > 1:
                triples.append(Triple(subject=subj, predicate="is", object=obj_text))

    return triples


class KnowledgeGraph:
    """In-memory knowledge graph for RAG with entity/relation extraction and graph search.

    Stores entities and relations as an adjacency structure. Supports
    multi-hop graph traversal and fusion with vector retrieval results.
    """

    def __init__(
        self,
        extract_fn: Callable[[str], str] | None = None,
    ):
        self._extract_fn = extract_fn
        # Adjacency: entity_id -> {related_entity_id: [Relation]}
        self._adjacency: dict[str, dict[str, list[Relation]]] = {}
        self._entities: dict[str, Entity] = {}
        self._triples: list[Triple] = []

    # ── Graph construction ────────────────────────────────────────────────────

    def add_triple(self, triple: Triple) -> None:
        """Add a single triple to the graph."""
        subj_id = triple.subject.lower().strip()
        obj_id = triple.object.lower().strip()

        # Register entities
        if subj_id not in self._entities:
            self._entities[subj_id] = Entity(name=triple.subject)
        else:
            self._entities[subj_id].mentions += 1

        if obj_id not in self._entities:
            self._entities[obj_id] = Entity(name=triple.object)
        else:
            self._entities[obj_id].mentions += 1

        # Add relation (bidirectional)
        rel = Relation(
            relation_type=triple.predicate,
            weight=triple.weight,
            source_doc=triple.source_doc,
        )

        if subj_id not in self._adjacency:
            self._adjacency[subj_id] = {}
        if obj_id not in self._adjacency[subj_id]:
            self._adjacency[subj_id][obj_id] = []
        self._adjacency[subj_id][obj_id].append(rel)

        # Reverse edge
        if obj_id not in self._adjacency:
            self._adjacency[obj_id] = {}
        if subj_id not in self._adjacency[obj_id]:
            self._adjacency[obj_id][subj_id] = []
        self._adjacency[obj_id][subj_id].append(
            Relation(relation_type=f"inverse_{triple.predicate}", weight=triple.weight)
        )

        self._triples.append(triple)

    def add_triples(self, triples: list[Triple]) -> None:
        """Add multiple triples to the graph."""
        for t in triples:
            self.add_triple(t)

    def add_document(self, text: str, source: str = "", chunk_size: int = 1000) -> list[Triple]:
        """Extract entities/relations from text and add to graph.

        Uses LLM extraction if extract_fn is provided, otherwise falls back
        to heuristic extraction.
        """
        all_triples: list[Triple] = []

        # Split into chunks for extraction
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

        for chunk in chunks:
            if self._extract_fn:
                prompt = EXTRACTION_PROMPT.format(text=chunk)
                raw = self._extract_fn(prompt)
                triples = _parse_triples(raw)
            else:
                triples = _heuristic_extract(chunk)

            for t in triples:
                t.source_doc = source
            all_triples.extend(triples)

        self.add_triples(all_triples)
        return all_triples

    # ── Graph search ──────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_hops: int = 2,
        top_k: int = 10,
    ) -> list[GraphSearchResult]:
        """Search the knowledge graph for entities related to the query.

        Performs BFS traversal from matching entities up to max_hops away.
        """
        # Find seed entities mentioned in query
        query_lower = query.lower()
        seed_entities = []
        for eid, entity in self._entities.items():
            if eid in query_lower or entity.name.lower() in query_lower:
                seed_entities.append(eid)

        if not seed_entities:
            # Fallback: find entities with word overlap
            query_words = set(query_lower.split())
            for eid, entity in self._entities.items():
                entity_words = set(eid.split())
                if query_words & entity_words:
                    seed_entities.append(eid)

        results = []
        for seed in seed_entities[:top_k]:
            result = self._bfs_search(seed, max_hops)
            results.append(result)

        # Sort by score (number of connections + mentions)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _bfs_search(self, seed: str, max_hops: int) -> GraphSearchResult:
        """BFS from seed entity, collecting related entities and paths."""
        visited: set[str] = {seed}
        queue: list[tuple[str, int, list[str]]] = [(seed, 0, [seed])]
        related: list[dict[str, Any]] = []
        collected_triples: list[Triple] = []
        all_paths: list[list[str]] = []

        while queue:
            current, hop, path = queue.pop(0)
            if hop >= max_hops:
                continue

            neighbors = self._adjacency.get(current, {})
            for neighbor_id, relations in neighbors.items():
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [neighbor_id]
                    queue.append((neighbor_id, hop + 1, new_path))
                    all_paths.append(new_path)

                    entity = self._entities.get(neighbor_id)
                    if entity:
                        related.append({
                            "entity": entity.name,
                            "relation": relations[0].relation_type if relations else "",
                            "hops": hop + 1,
                            "mentions": entity.mentions,
                        })

                # Collect triples for this edge
                for rel in relations:
                    collected_triples.append(Triple(
                        subject=self._entities.get(current, Entity(name=current)).name,
                        predicate=rel.relation_type,
                        object=self._entities.get(neighbor_id, Entity(name=neighbor_id)).name,
                    ))

        entity = self._entities.get(seed, Entity(name=seed))
        score = len(related) * 0.5 + entity.mentions * 0.3

        return GraphSearchResult(
            entity=entity.name,
            related_entities=related,
            paths=all_paths,
            triples=collected_triples,
            score=score,
            hops=min(max_hops, max((r["hops"] for r in related), default=0)),
        )

    def get_neighbors(self, entity_name: str) -> list[dict[str, Any]]:
        """Get direct neighbors of an entity."""
        eid = entity_name.lower().strip()
        neighbors = self._adjacency.get(eid, {})
        result = []
        for nid, relations in neighbors.items():
            entity = self._entities.get(nid)
            if entity:
                result.append({
                    "entity": entity.name,
                    "relations": [r.relation_type for r in relations],
                    "mentions": entity.mentions,
                })
        return result

    def get_entity(self, name: str) -> Entity | None:
        """Look up an entity by name."""
        return self._entities.get(name.lower().strip())

    # ── Fusion with vector retrieval ──────────────────────────────────────────

    def fuse(
        self,
        graph_results: list[GraphSearchResult],
        vector_results: list,
        graph_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Fuse graph search results with vector retrieval results.

        Returns a unified ranked list combining graph structural knowledge
        with semantic vector similarity.
        """
        fused: dict[str, dict[str, Any]] = {}

        # Score graph results
        max_graph = max((r.score for r in graph_results), default=1.0) or 1.0
        for r in graph_results:
            key = r.entity.lower()
            fused[key] = {
                "entity": r.entity,
                "graph_score": (r.score / max_graph) * graph_weight,
                "vector_score": 0.0,
                "combined_score": 0.0,
                "source": "graph",
                "triples": [t.to_dict() for t in r.triples[:5]],
            }

        # Score vector results
        for r in vector_results:
            text_preview = r.chunk.text[:100] if hasattr(r, 'chunk') else str(r)[:100]
            key = text_preview.lower()[:50]
            if key not in fused:
                fused[key] = {
                    "text": text_preview,
                    "graph_score": 0.0,
                    "vector_score": r.score * vector_weight if hasattr(r, 'score') else 0.0,
                    "combined_score": 0.0,
                    "source": "vector",
                }
            else:
                fused[key]["vector_score"] = r.score * vector_weight if hasattr(r, 'score') else 0.0

        # Compute combined scores
        for item in fused.values():
            item["combined_score"] = item["graph_score"] + item["vector_score"]

        # Sort by combined score
        ranked = sorted(fused.values(), key=lambda x: x["combined_score"], reverse=True)
        return ranked

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def triple_count(self) -> int:
        return len(self._triples)

    @property
    def entities(self) -> list[Entity]:
        return list(self._entities.values())

    @property
    def triples(self) -> list[Triple]:
        return list(self._triples)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable graph representation."""
        return {
            "entity_count": self.entity_count,
            "triple_count": self.triple_count,
            "entities": [
                {"name": e.name, "type": e.entity_type, "mentions": e.mentions}
                for e in self._entities.values()
            ],
            "triples": [t.to_dict() for t in self._triples],
        }

    def summary(self) -> str:
        """Human-readable graph summary."""
        top_entities = sorted(
            self._entities.values(), key=lambda e: e.mentions, reverse=True
        )[:10]
        lines = [f"KnowledgeGraph: {self.entity_count} entities, {self.triple_count} triples"]
        if top_entities:
            lines.append("  Top entities:")
            for e in top_entities:
                neighbors = len(self._adjacency.get(e.id, {}))
                lines.append(f"    {e.name} (mentions={e.mentions}, neighbors={neighbors})")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all graph data."""
        self._adjacency.clear()
        self._entities.clear()
        self._triples.clear()
