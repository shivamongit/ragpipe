"""Community detection for knowledge graph summarization.

Implements label-propagation clustering and optional LLM-powered community
summarization, inspired by Microsoft GraphRAG.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ragpipe.graph.entities import KnowledgeGraph


@dataclass
class Community:
    """A group of closely-connected entities in a knowledge graph."""

    community_id: int
    entities: list[str] = field(default_factory=list)  # entity ids
    summary: str = ""
    level: int = 0
    size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.size:
            self.size = len(self.entities)

    def __repr__(self) -> str:
        return (
            f"Community(id={self.community_id}, size={self.size}, "
            f"summary='{self.summary[:60]}...')" if len(self.summary) > 60
            else f"Community(id={self.community_id}, size={self.size}, summary='{self.summary}')"
        )


_SUMMARIZE_PROMPT = """Summarize the following group of related entities and their relationships in 1-2 sentences.
Focus on the main theme that connects these entities.

Entities:
{entities}

Relationships:
{relationships}

Summary:"""


class CommunityDetector:
    """Detect communities in a knowledge graph using label propagation.

    When *summarize_fn* is provided it is called with a prompt string and
    should return a text summary.  Otherwise a heuristic summary is generated
    from the entity names and types.

    Example::

        detector = CommunityDetector(min_community_size=3)
        communities = detector.detect(graph)
        for c in communities:
            print(c.community_id, c.size, c.summary)
    """

    def __init__(
        self,
        summarize_fn: Optional[Callable[[str], str]] = None,
        min_community_size: int = 2,
        max_iterations: int = 50,
    ) -> None:
        self._summarize_fn = summarize_fn
        self.min_community_size = min_community_size
        self.max_iterations = max_iterations

    # -- public API ---------------------------------------------------------

    def detect(self, graph: KnowledgeGraph) -> list[Community]:
        """Detect communities and return them with heuristic summaries."""
        if graph.entity_count == 0:
            return []

        labels = self._label_propagation(graph)
        communities = self._build_communities(labels)

        # Merge small communities into nearest larger community
        communities = self._merge_small(graph, communities)

        # Generate summaries
        for comm in communities:
            comm.summary = self._summarize_community(graph, comm.entities)
        return communities

    async def adetect(self, graph: KnowledgeGraph) -> list[Community]:
        """Async detect. Override for native async; default wraps sync in a thread."""
        return await asyncio.to_thread(self.detect, graph)

    def summarize_all(
        self,
        graph: KnowledgeGraph,
        communities: list[Community],
    ) -> list[Community]:
        """(Re-)generate summaries for all *communities*."""
        for comm in communities:
            comm.summary = self._summarize_community(graph, comm.entities)
        return communities

    # -- label propagation --------------------------------------------------

    def _label_propagation(self, graph: KnowledgeGraph) -> dict[str, int]:
        """Run label propagation and return ``{entity_id: community_label}``."""
        entity_ids = list(graph.entities.keys())
        labels: dict[str, int] = {eid: i for i, eid in enumerate(entity_ids)}

        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for rel in graph.relationships:
            adjacency[rel.source].append((rel.target, rel.weight))
            adjacency[rel.target].append((rel.source, rel.weight))

        for _ in range(self.max_iterations):
            changed = False
            order = list(entity_ids)
            random.shuffle(order)
            for eid in order:
                neighbors = adjacency.get(eid, [])
                if not neighbors:
                    continue
                # Weighted vote
                votes: dict[int, float] = defaultdict(float)
                for neighbor_id, weight in neighbors:
                    votes[labels[neighbor_id]] += weight
                if not votes:
                    continue
                best_label = max(votes, key=lambda lbl: votes[lbl])
                if labels[eid] != best_label:
                    labels[eid] = best_label
                    changed = True
            if not changed:
                break
        return labels

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _build_communities(labels: dict[str, int]) -> list[Community]:
        """Group entity ids by label into :class:`Community` objects."""
        groups: dict[int, list[str]] = defaultdict(list)
        for eid, label in labels.items():
            groups[label].append(eid)

        communities: list[Community] = []
        for idx, (_, members) in enumerate(sorted(groups.items())):
            communities.append(Community(
                community_id=idx,
                entities=members,
                size=len(members),
            ))
        return communities

    def _merge_small(
        self,
        graph: KnowledgeGraph,
        communities: list[Community],
    ) -> list[Community]:
        """Merge communities smaller than *min_community_size* into the nearest larger one."""
        large: list[Community] = []
        small: list[Community] = []
        for c in communities:
            (large if c.size >= self.min_community_size else small).append(c)

        if not large:
            # Nothing to merge into; return all as-is
            return communities

        for sc in small:
            best_comm = self._find_nearest(graph, sc, large)
            best_comm.entities.extend(sc.entities)
            best_comm.size = len(best_comm.entities)

        # Re-number ids
        for idx, comm in enumerate(large):
            comm.community_id = idx
        return large

    @staticmethod
    def _find_nearest(
        graph: KnowledgeGraph,
        small: Community,
        large: list[Community],
    ) -> Community:
        """Find the large community most connected to *small*."""
        small_set = set(small.entities)
        best: Community = large[0]
        best_connections = 0
        for comm in large:
            comm_set = set(comm.entities)
            connections = sum(
                1
                for rel in graph.relationships
                if (rel.source in small_set and rel.target in comm_set)
                or (rel.target in small_set and rel.source in comm_set)
            )
            if connections > best_connections:
                best_connections = connections
                best = comm
        return best

    def _summarize_community(self, graph: KnowledgeGraph, entity_ids: list[str]) -> str:
        """Generate a summary for a community of entities."""
        entities_text = []
        for eid in entity_ids:
            entity = graph.get_entity(eid)
            if entity is not None:
                desc = f" — {entity.description}" if entity.description else ""
                entities_text.append(f"- {entity.name} ({entity.entity_type}){desc}")

        rels_text = []
        id_set = set(entity_ids)
        for rel in graph.relationships:
            if rel.source in id_set and rel.target in id_set:
                src = graph.get_entity(rel.source)
                tgt = graph.get_entity(rel.target)
                src_name = src.name if src else rel.source
                tgt_name = tgt.name if tgt else rel.target
                rels_text.append(f"- {src_name} --[{rel.relation_type}]--> {tgt_name}")

        if self._summarize_fn:
            prompt = _SUMMARIZE_PROMPT.format(
                entities="\n".join(entities_text) or "(none)",
                relationships="\n".join(rels_text) or "(none)",
            )
            try:
                return self._summarize_fn(prompt).strip()
            except Exception:
                pass  # Fall through to heuristic

        # Heuristic summary
        if not entities_text:
            return ""
        names = [
            graph.get_entity(eid).name
            for eid in entity_ids
            if graph.get_entity(eid) is not None
        ]
        types = {
            graph.get_entity(eid).entity_type
            for eid in entity_ids
            if graph.get_entity(eid) is not None
        }
        type_str = ", ".join(sorted(types))
        if len(names) <= 3:
            return f"Community of {type_str} entities: {', '.join(names)}."
        return (
            f"Community of {len(names)} entities ({type_str}) including "
            f"{', '.join(names[:3])}, and others."
        )
