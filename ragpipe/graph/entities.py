"""Core graph data structures and entity extraction."""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A named entity extracted from text."""

    name: str
    entity_type: str  # PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, EVENT
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    source_doc_id: str = ""

    @property
    def id(self) -> str:
        return f"{self.entity_type}:{self.name.lower().replace(' ', '_')}"

    def __repr__(self) -> str:
        return f"Entity(id='{self.id}', name='{self.name}', type='{self.entity_type}')"


@dataclass
class Relationship:
    """A directed relationship between two entities."""

    source: str  # entity id
    target: str  # entity id
    relation_type: str  # WORKS_FOR, LOCATED_IN, RELATED_TO, DEPENDS_ON, etc.
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    source_doc_id: str = ""

    def __repr__(self) -> str:
        return (
            f"Relationship(source='{self.source}', target='{self.target}', "
            f"type='{self.relation_type}', weight={self.weight:.2f})"
        )


# ---------------------------------------------------------------------------
# In-memory knowledge graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """In-memory knowledge graph with entity and relationship storage.

    Example::

        graph = KnowledgeGraph()
        graph.add_entity(Entity(name="Alice", entity_type="PERSON"))
        graph.add_entity(Entity(name="Acme", entity_type="ORGANIZATION"))
        graph.add_relationship(Relationship(
            source="PERSON:alice", target="ORGANIZATION:acme",
            relation_type="WORKS_FOR",
        ))
        print(graph.stats())
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self._adjacency: dict[str, list[str]] = defaultdict(list)

    # -- mutators -----------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph, merging descriptions on duplicate ids."""
        existing = self.entities.get(entity.id)
        if existing is not None:
            if entity.description and entity.description not in existing.description:
                existing.description = (
                    f"{existing.description}; {entity.description}"
                    if existing.description
                    else entity.description
                )
            existing.properties.update(entity.properties)
        else:
            self.entities[entity.id] = entity

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship and update the adjacency index."""
        self.relationships.append(rel)
        if rel.target not in self._adjacency[rel.source]:
            self._adjacency[rel.source].append(rel.target)
        if rel.source not in self._adjacency[rel.target]:
            self._adjacency[rel.target].append(rel.source)

    # -- queries ------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Return an entity by id, or *None* if not found."""
        return self.entities.get(entity_id)

    def get_neighbors(self, entity_id: str, max_hops: int = 1) -> list[Entity]:
        """Return entities reachable within *max_hops* from *entity_id*."""
        visited: set[str] = {entity_id}
        frontier: set[str] = {entity_id}
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for eid in frontier:
                for neighbor in self._adjacency.get(eid, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(entity_id)
        return [self.entities[eid] for eid in visited if eid in self.entities]

    def get_relationships_for(self, entity_id: str) -> list[Relationship]:
        """Return all relationships involving *entity_id* as source or target."""
        return [
            r for r in self.relationships
            if r.source == entity_id or r.target == entity_id
        ]

    # -- subgraph / merge ---------------------------------------------------

    def subgraph(self, entity_ids: list[str]) -> KnowledgeGraph:
        """Return a new graph containing only the specified entities and their inter-relationships."""
        sub = KnowledgeGraph()
        id_set = set(entity_ids)
        for eid in entity_ids:
            entity = self.entities.get(eid)
            if entity is not None:
                sub.add_entity(entity)
        for rel in self.relationships:
            if rel.source in id_set and rel.target in id_set:
                sub.add_relationship(rel)
        return sub

    def merge(self, other: KnowledgeGraph) -> None:
        """Merge another graph into this one."""
        for entity in other.entities.values():
            self.add_entity(entity)
        for rel in other.relationships:
            self.add_relationship(rel)

    # -- serialization / stats ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a plain dictionary."""
        return {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "description": e.description,
                    "properties": e.properties,
                    "source_doc_id": e.source_doc_id,
                }
                for e in self.entities.values()
            ],
            "relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "weight": r.weight,
                    "properties": r.properties,
                    "source_doc_id": r.source_doc_id,
                }
                for r in self.relationships
            ],
        }

    def stats(self) -> dict[str, Any]:
        """Return summary statistics for the graph."""
        type_counts: dict[str, int] = defaultdict(int)
        for e in self.entities.values():
            type_counts[e.entity_type] += 1
        return {
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "entity_types": dict(type_counts),
        }

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    def __repr__(self) -> str:
        return f"KnowledgeGraph(entities={self.entity_count}, relationships={self.relationship_count})"


# ---------------------------------------------------------------------------
# Entity / relationship extraction
# ---------------------------------------------------------------------------

_DEFAULT_ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "CONCEPT",
    "TECHNOLOGY",
    "EVENT",
]

_EXTRACTION_PROMPT = """Extract all named entities and relationships from the following text.

Return ONLY valid JSON with this schema:
{{
  "entities": [
    {{"name": "...", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|TECHNOLOGY|EVENT", "description": "..."}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "type": "RELATIONSHIP_TYPE"}}
  ]
}}

Text:
{text}
"""

# Heuristic patterns
_ORG_SUFFIXES = re.compile(
    r"\b(?:Inc|Corp|LLC|Ltd|Co|Company|Foundation|Institute|University|Association)\b",
    re.IGNORECASE,
)
_PERSON_PREFIXES = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|President|CEO|CTO|Director|Manager|Senator|Judge)\b\.?",
    re.IGNORECASE,
)
_TECH_KEYWORDS = re.compile(
    r"\b(?:API|SDK|Python|Java|JavaScript|TypeScript|Rust|Go|Docker|Kubernetes|AWS|Azure|GCP|SQL|NoSQL|GraphQL|REST|HTTP|TCP|GPU|CPU|ML|AI|NLP|LLM)\b",
    re.IGNORECASE,
)
_LOCATION_KEYWORDS = re.compile(
    r"\b(?:City|State|Country|Street|Avenue|Road|River|Mountain|Ocean|Lake|Island)\b",
    re.IGNORECASE,
)
_CAPITALIZED_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


class EntityExtractor:
    """Extract entities and relationships from text using an LLM or heuristics.

    When *extract_fn* is provided it is called with a prompt string and should
    return the raw LLM response text.  When omitted, a regex-based heuristic
    extracts capitalized phrases and classifies them by context keywords.

    Example::

        extractor = EntityExtractor()
        entities, relationships = extractor.extract("Dr. Alice works at Acme Corp.")
    """

    def __init__(
        self,
        extract_fn: Optional[Callable[[str], str]] = None,
        entity_types: Optional[list[str]] = None,
    ) -> None:
        self._extract_fn = extract_fn
        self.entity_types = entity_types or list(_DEFAULT_ENTITY_TYPES)

    # -- public API ---------------------------------------------------------

    def extract(self, text: str, doc_id: str = "") -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from *text* (sync)."""
        if self._extract_fn is not None:
            prompt = _EXTRACTION_PROMPT.format(text=text[:4000])
            raw = self._extract_fn(prompt)
            entities, rels = self._parse_extraction(raw, doc_id)
            if entities:
                return entities, rels
        return self._heuristic_extract(text, doc_id)

    async def aextract(self, text: str, doc_id: str = "") -> tuple[list[Entity], list[Relationship]]:
        """Async extract. Override for native async; default wraps sync in a thread."""
        return await asyncio.to_thread(self.extract, text, doc_id)

    # -- LLM response parsing -----------------------------------------------

    def _parse_extraction(self, raw: str, doc_id: str) -> tuple[list[Entity], list[Relationship]]:
        """Parse structured JSON from an LLM response into entities and relationships."""
        raw = raw.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return [], []
        try:
            data = json.loads(json_match.group())
        except (json.JSONDecodeError, ValueError):
            return [], []

        entities: list[Entity] = []
        entity_name_map: dict[str, Entity] = {}
        for item in data.get("entities", []):
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "CONCEPT")).upper().strip()
            if not name:
                continue
            if etype not in self.entity_types:
                etype = "CONCEPT"
            entity = Entity(
                name=name,
                entity_type=etype,
                description=str(item.get("description", "")),
                source_doc_id=doc_id,
            )
            entities.append(entity)
            entity_name_map[name.lower()] = entity

        relationships: list[Relationship] = []
        for item in data.get("relationships", []):
            src_name = str(item.get("source", "")).strip().lower()
            tgt_name = str(item.get("target", "")).strip().lower()
            rel_type = str(item.get("type", "RELATED_TO")).upper().strip()
            src_entity = entity_name_map.get(src_name)
            tgt_entity = entity_name_map.get(tgt_name)
            if src_entity is None or tgt_entity is None:
                continue
            relationships.append(Relationship(
                source=src_entity.id,
                target=tgt_entity.id,
                relation_type=rel_type,
                source_doc_id=doc_id,
            ))
        return entities, relationships

    # -- heuristic (no-LLM) extraction --------------------------------------

    def _heuristic_extract(self, text: str, doc_id: str) -> tuple[list[Entity], list[Relationship]]:
        """Regex-based entity extraction and sentence co-occurrence relationships."""
        entities: list[Entity] = []
        seen: set[str] = set()

        # Find capitalized multi-word phrases (proper nouns)
        for match in _CAPITALIZED_PHRASE.finditer(text):
            name = match.group(1).strip()
            if name.lower() in seen or len(name) < 3:
                continue
            seen.add(name.lower())
            etype = self._classify_entity(name, text)
            entities.append(Entity(name=name, entity_type=etype, source_doc_id=doc_id))

        # Find technology keywords
        for match in _TECH_KEYWORDS.finditer(text):
            name = match.group(0).strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            entities.append(Entity(name=name, entity_type="TECHNOLOGY", source_doc_id=doc_id))

        # Build relationships from sentence co-occurrence
        relationships = self._cooccurrence_relationships(text, entities, doc_id)
        return entities, relationships

    def _classify_entity(self, name: str, context: str) -> str:
        """Classify an entity type by context keywords surrounding *name*."""
        # Check the entity name itself first (strongest signal)
        if _ORG_SUFFIXES.search(name):
            return "ORGANIZATION"
        if _TECH_KEYWORDS.search(name):
            return "TECHNOLOGY"

        # Check surrounding context (100 chars around the name)
        idx = context.find(name)
        window = context[max(0, idx - 100): idx + len(name) + 100] if idx >= 0 else ""

        if _PERSON_PREFIXES.search(window):
            # Verify the person prefix is adjacent to this entity, not another
            for m in _PERSON_PREFIXES.finditer(window):
                after_prefix = window[m.end():].lstrip()
                if after_prefix.startswith(name):
                    return "PERSON"
        if _LOCATION_KEYWORDS.search(name):
            return "LOCATION"
        if _ORG_SUFFIXES.search(window):
            return "ORGANIZATION"
        if _LOCATION_KEYWORDS.search(window):
            return "LOCATION"
        return "CONCEPT"

    @staticmethod
    def _cooccurrence_relationships(
        text: str,
        entities: list[Entity],
        doc_id: str,
    ) -> list[Relationship]:
        """Create RELATED_TO relationships for entities co-occurring in the same sentence."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        relationships: list[Relationship] = []
        seen_pairs: set[tuple[str, str]] = set()

        for sentence in sentences:
            sentence_lower = sentence.lower()
            present = [e for e in entities if e.name.lower() in sentence_lower]
            for i, src in enumerate(present):
                for tgt in present[i + 1:]:
                    pair = (src.id, tgt.id) if src.id < tgt.id else (tgt.id, src.id)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    relationships.append(Relationship(
                        source=src.id,
                        target=tgt.id,
                        relation_type="RELATED_TO",
                        source_doc_id=doc_id,
                    ))
        return relationships
