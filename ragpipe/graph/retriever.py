"""Graph-enhanced retrieval combining vector search with knowledge graph traversal."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ragpipe.graph.community import Community
from ragpipe.graph.entities import Entity, KnowledgeGraph, Relationship


@dataclass
class GraphRAGResult:
    """Result from graph-enhanced retrieval."""

    answer: str
    entities_used: list[Entity] = field(default_factory=list)
    relationships_used: list[Relationship] = field(default_factory=list)
    communities_used: list[Community] = field(default_factory=list)
    subgraph: Optional[KnowledgeGraph] = None
    confidence: float = 0.0
    strategy: str = "hybrid"  # "local", "global", "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.answer[:80] + "..." if len(self.answer) > 80 else self.answer
        return (
            f"GraphRAGResult(strategy='{self.strategy}', confidence={self.confidence:.2f}, "
            f"entities={len(self.entities_used)}, answer='{preview}')"
        )


# ---------------------------------------------------------------------------
# Internal prompts
# ---------------------------------------------------------------------------

_LOCAL_PROMPT = """Answer the following question using ONLY the provided knowledge graph context.
If you cannot answer from the context, say so.

Question: {query}

Entities:
{entities}

Relationships:
{relationships}

Answer:"""

_GLOBAL_PROMPT = """Answer the following question using the community summaries below.
Synthesize information across communities to give a comprehensive answer.

Question: {query}

Community Summaries:
{summaries}

Answer:"""

_HYBRID_PROMPT = """Answer the following question using the provided context from both
knowledge graph traversal and document retrieval.

Question: {query}

Knowledge Graph Context:
{graph_context}

Retrieved Documents:
{doc_context}

Answer:"""


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------

class GraphRetriever:
    """Graph-enhanced retrieval combining vector search with knowledge graph traversal.

    Supports three retrieval strategies:

    - **local**: Extract entities from the query, traverse the graph, and
      generate an answer from the subgraph context.
    - **global**: Use pre-computed community summaries to answer broad queries
      via map-reduce over communities.
    - **hybrid**: Combine local graph traversal with vector retrieval results.

    Example::

        retriever = GraphRetriever(
            graph=kg,
            communities=communities,
            generate_fn=my_llm,
            strategy="hybrid",
        )
        result = retriever.retrieve("What projects does Alice work on?")
        print(result.answer, result.confidence)
    """

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        communities: Optional[list[Community]] = None,
        vector_retrieve_fn: Optional[Callable[[str, int], list[Any]]] = None,
        generate_fn: Optional[Callable[[str], str]] = None,
        entity_extract_fn: Optional[Callable[[str], list[str]]] = None,
        max_hops: int = 2,
        strategy: str = "hybrid",
    ) -> None:
        self.graph = graph or KnowledgeGraph()
        self.communities = communities or []
        self._vector_retrieve_fn = vector_retrieve_fn
        self._generate_fn = generate_fn
        self._entity_extract_fn = entity_extract_fn
        self.max_hops = max_hops
        self.strategy = strategy

    # -- public API ---------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> GraphRAGResult:
        """Retrieve an answer using the configured strategy (sync)."""
        strategies = {
            "local": self._local_search,
            "global": self._global_search,
            "hybrid": self._hybrid_search,
        }
        search_fn = strategies.get(self.strategy, self._hybrid_search)
        return search_fn(query, top_k)

    async def aretrieve(self, query: str, top_k: int = 5) -> GraphRAGResult:
        """Async retrieve. Override for native async; default wraps sync in a thread."""
        return await asyncio.to_thread(self.retrieve, query, top_k)

    # -- local search -------------------------------------------------------

    def _local_search(self, query: str, top_k: int) -> GraphRAGResult:
        """Extract entities from the query, traverse the graph, and generate."""
        seed_ids = self._extract_query_entities(query)
        subgraph = self._traverse_graph(seed_ids, self.max_hops)
        entities = list(subgraph.entities.values())
        entities = self._rank_entities(entities, query)[:top_k]
        entity_ids = [e.id for e in entities]
        rels = [
            r for r in subgraph.relationships
            if r.source in set(entity_ids) or r.target in set(entity_ids)
        ]

        answer = self._generate_local(query, entities, rels)
        confidence = min(1.0, len(entities) / max(top_k, 1))
        return GraphRAGResult(
            answer=answer,
            entities_used=entities,
            relationships_used=rels,
            subgraph=subgraph,
            confidence=confidence,
            strategy="local",
        )

    # -- global search ------------------------------------------------------

    def _global_search(self, query: str, top_k: int) -> GraphRAGResult:
        """Answer broad queries using community summaries (map-reduce)."""
        if not self.communities:
            return GraphRAGResult(
                answer="No community summaries available for global search.",
                strategy="global",
            )

        ranked = self._rank_communities(query)[:top_k]
        answer = self._generate_global(query, ranked)
        confidence = min(1.0, len(ranked) / max(top_k, 1))
        return GraphRAGResult(
            answer=answer,
            communities_used=ranked,
            confidence=confidence,
            strategy="global",
        )

    # -- hybrid search ------------------------------------------------------

    def _hybrid_search(self, query: str, top_k: int) -> GraphRAGResult:
        """Combine local graph traversal with vector retrieval."""
        # Graph traversal component
        seed_ids = self._extract_query_entities(query)
        subgraph = self._traverse_graph(seed_ids, self.max_hops)
        entities = list(subgraph.entities.values())
        entities = self._rank_entities(entities, query)[:top_k]
        entity_ids = [e.id for e in entities]
        rels = [
            r for r in subgraph.relationships
            if r.source in set(entity_ids) or r.target in set(entity_ids)
        ]

        # Vector retrieval component
        doc_context = ""
        if self._vector_retrieve_fn is not None:
            try:
                results = self._vector_retrieve_fn(query, top_k)
                doc_context = self._format_retrieval_results(results)
            except Exception:
                doc_context = ""

        answer = self._generate_hybrid(query, entities, rels, doc_context)
        confidence = min(1.0, (len(entities) + (1 if doc_context else 0)) / max(top_k, 1))
        return GraphRAGResult(
            answer=answer,
            entities_used=entities,
            relationships_used=rels,
            subgraph=subgraph,
            confidence=confidence,
            strategy="hybrid",
        )

    # -- entity extraction from query ---------------------------------------

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity ids from a query string."""
        if self._entity_extract_fn is not None:
            try:
                names = self._entity_extract_fn(query)
                return self._match_entity_names(names)
            except Exception:
                pass

        # Heuristic: find capitalized phrases and single capitalized words
        candidates: list[str] = []
        # Multi-word capitalized phrases
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
            candidates.append(match.group(1))
        # Single capitalized words (not at sentence start) or all-caps words
        tokens = query.split()
        for i, token in enumerate(tokens):
            clean = re.sub(r"[^\w]", "", token)
            if not clean:
                continue
            if clean[0].isupper() and i > 0 and len(clean) > 1:
                candidates.append(clean)
            elif clean.isupper() and len(clean) > 1:
                candidates.append(clean)

        return self._match_entity_names(candidates)

    def _match_entity_names(self, names: list[str]) -> list[str]:
        """Match candidate names to entity ids in the graph."""
        matched: list[str] = []
        name_lower_map = {e.name.lower(): e.id for e in self.graph.entities.values()}
        for name in names:
            eid = name_lower_map.get(name.lower())
            if eid is not None:
                matched.append(eid)
        return matched

    # -- graph traversal ----------------------------------------------------

    def _traverse_graph(self, seed_entities: list[str], max_hops: int) -> KnowledgeGraph:
        """Build a subgraph by traversing *max_hops* from seed entities."""
        if not seed_entities:
            return KnowledgeGraph()

        visited: set[str] = set(seed_entities)
        frontier: set[str] = set(seed_entities)
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for eid in frontier:
                for neighbor_id in self.graph._adjacency.get(eid, []):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        return self.graph.subgraph(list(visited))

    # -- ranking ------------------------------------------------------------

    def _rank_entities(self, entities: list[Entity], query: str) -> list[Entity]:
        """Rank entities by relevance to the query using word overlap."""
        query_words = set(query.lower().split())
        if not query_words:
            return entities

        def _score(entity: Entity) -> float:
            name_words = set(entity.name.lower().split())
            desc_words = set(entity.description.lower().split()) if entity.description else set()
            all_words = name_words | desc_words
            if not all_words:
                return 0.0
            overlap = len(query_words & all_words)
            # Boost for exact name match
            name_overlap = len(query_words & name_words)
            return overlap + name_overlap * 2.0

        return sorted(entities, key=_score, reverse=True)

    def _rank_communities(self, query: str) -> list[Community]:
        """Rank communities by relevance of their summaries to the query."""
        query_words = set(query.lower().split())
        if not query_words:
            return list(self.communities)

        def _score(community: Community) -> float:
            summary_words = set(community.summary.lower().split())
            if not summary_words:
                return 0.0
            return len(query_words & summary_words) / len(query_words)

        return sorted(self.communities, key=_score, reverse=True)

    # -- generation helpers -------------------------------------------------

    def _generate_local(
        self,
        query: str,
        entities: list[Entity],
        rels: list[Relationship],
    ) -> str:
        """Generate an answer from local graph context."""
        entities_text = self._format_entities(entities)
        rels_text = self._format_relationships(rels)

        if self._generate_fn is not None:
            prompt = _LOCAL_PROMPT.format(
                query=query,
                entities=entities_text or "(none)",
                relationships=rels_text or "(none)",
            )
            try:
                return self._generate_fn(prompt).strip()
            except Exception:
                pass

        # Heuristic: concatenate entity information
        if not entities:
            return "No relevant entities found in the knowledge graph."
        parts = [f"{e.name} ({e.entity_type}): {e.description}" if e.description else f"{e.name} ({e.entity_type})" for e in entities]
        return "Based on the knowledge graph: " + "; ".join(parts) + "."

    def _generate_global(self, query: str, communities: list[Community]) -> str:
        """Generate an answer from community summaries."""
        summaries = "\n".join(
            f"[Community {c.community_id}] {c.summary}" for c in communities
        )

        if self._generate_fn is not None:
            prompt = _GLOBAL_PROMPT.format(query=query, summaries=summaries or "(none)")
            try:
                return self._generate_fn(prompt).strip()
            except Exception:
                pass

        # Heuristic: return concatenated summaries
        if not communities:
            return "No relevant communities found."
        return "Based on community analysis: " + " ".join(c.summary for c in communities if c.summary)

    def _generate_hybrid(
        self,
        query: str,
        entities: list[Entity],
        rels: list[Relationship],
        doc_context: str,
    ) -> str:
        """Generate an answer combining graph and document context."""
        graph_context = self._format_entities(entities)
        if rels:
            graph_context += "\n" + self._format_relationships(rels)

        if self._generate_fn is not None:
            prompt = _HYBRID_PROMPT.format(
                query=query,
                graph_context=graph_context or "(none)",
                doc_context=doc_context or "(none)",
            )
            try:
                return self._generate_fn(prompt).strip()
            except Exception:
                pass

        # Heuristic: combine available context
        parts: list[str] = []
        if entities:
            names = ", ".join(e.name for e in entities[:5])
            parts.append(f"Related entities: {names}")
        if doc_context:
            parts.append(f"Documents: {doc_context[:200]}")
        if not parts:
            return "No relevant information found."
        return "Based on hybrid search: " + ". ".join(parts) + "."

    # -- formatting helpers -------------------------------------------------

    @staticmethod
    def _format_entities(entities: list[Entity]) -> str:
        """Format entities as readable text for prompts."""
        lines: list[str] = []
        for e in entities:
            desc = f" — {e.description}" if e.description else ""
            lines.append(f"- {e.name} ({e.entity_type}){desc}")
        return "\n".join(lines)

    @staticmethod
    def _format_relationships(rels: list[Relationship]) -> str:
        """Format relationships as readable text for prompts."""
        lines: list[str] = []
        for r in rels:
            lines.append(f"- {r.source} --[{r.relation_type}]--> {r.target}")
        return "\n".join(lines)

    @staticmethod
    def _format_retrieval_results(results: list[Any]) -> str:
        """Format vector retrieval results as readable text."""
        lines: list[str] = []
        for r in results:
            text = getattr(r, "text", None) or getattr(r, "content", None) or str(r)
            if hasattr(r, "chunk"):
                text = getattr(r.chunk, "text", text)
            lines.append(f"- {str(text)[:300]}")
        return "\n".join(lines)
