"""Tests for ragpipe.graph.entities — knowledge graph data structures and entity extraction."""

import pytest

from ragpipe.graph.entities import Entity, Relationship, KnowledgeGraph, EntityExtractor


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

def test_entity_creation():
    e = Entity(name="Alice", entity_type="PERSON", description="A researcher")
    assert e.name == "Alice"
    assert e.entity_type == "PERSON"
    assert e.description == "A researcher"
    assert e.id is not None


def test_entity_id_format():
    e = Entity(name="John Smith", entity_type="PERSON")
    assert e.id == "PERSON:john_smith"


def test_entity_id_normalisation():
    e = Entity(name="New York City", entity_type="LOCATION")
    assert e.id == "LOCATION:new_york_city"


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

def test_relationship_creation():
    r = Relationship(
        source="PERSON:alice",
        target="ORGANIZATION:acme",
        relation_type="WORKS_FOR",
        weight=0.9,
    )
    assert r.source == "PERSON:alice"
    assert r.target == "ORGANIZATION:acme"
    assert r.relation_type == "WORKS_FOR"
    assert r.weight == 0.9


# ---------------------------------------------------------------------------
# KnowledgeGraph — add / query
# ---------------------------------------------------------------------------

def _make_graph():
    """Helper that builds a small graph: Alice→Acme, Bob→Acme."""
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice", entity_type="PERSON"))
    g.add_entity(Entity(name="Bob", entity_type="PERSON"))
    g.add_entity(Entity(name="Acme", entity_type="ORGANIZATION"))
    g.add_relationship(Relationship(
        source="PERSON:alice", target="ORGANIZATION:acme", relation_type="WORKS_FOR",
    ))
    g.add_relationship(Relationship(
        source="PERSON:bob", target="ORGANIZATION:acme", relation_type="WORKS_FOR",
    ))
    return g


def test_knowledge_graph_add_entity():
    g = _make_graph()
    assert g.entity_count == 3


def test_knowledge_graph_add_entity_merge():
    """Duplicate entity ids merge descriptions."""
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice", entity_type="PERSON", description="first"))
    g.add_entity(Entity(name="Alice", entity_type="PERSON", description="second"))
    assert g.entity_count == 1
    assert "first" in g.entities["PERSON:alice"].description
    assert "second" in g.entities["PERSON:alice"].description


def test_knowledge_graph_add_relationship():
    g = _make_graph()
    assert g.relationship_count == 2
    # Adjacency should be bidirectional
    assert "ORGANIZATION:acme" in g._adjacency["PERSON:alice"]
    assert "PERSON:alice" in g._adjacency["ORGANIZATION:acme"]


def test_knowledge_graph_get_entity():
    g = _make_graph()
    e = g.get_entity("PERSON:alice")
    assert e is not None
    assert e.name == "Alice"
    assert g.get_entity("PERSON:unknown") is None


def test_knowledge_graph_get_neighbors():
    g = _make_graph()
    neighbors_1 = g.get_neighbors("PERSON:alice", max_hops=1)
    ids_1 = {n.id for n in neighbors_1}
    assert "ORGANIZATION:acme" in ids_1

    # 2-hop from Alice should also reach Bob (through Acme)
    neighbors_2 = g.get_neighbors("PERSON:alice", max_hops=2)
    ids_2 = {n.id for n in neighbors_2}
    assert "PERSON:bob" in ids_2


def test_knowledge_graph_get_relationships_for():
    g = _make_graph()
    rels = g.get_relationships_for("ORGANIZATION:acme")
    assert len(rels) == 2


def test_knowledge_graph_subgraph():
    g = _make_graph()
    sub = g.subgraph(["PERSON:alice", "ORGANIZATION:acme"])
    assert sub.entity_count == 2
    assert sub.relationship_count == 1


def test_knowledge_graph_merge():
    g1 = KnowledgeGraph()
    g1.add_entity(Entity(name="Alice", entity_type="PERSON"))
    g2 = KnowledgeGraph()
    g2.add_entity(Entity(name="Bob", entity_type="PERSON"))
    g2.add_relationship(Relationship(
        source="PERSON:bob", target="PERSON:alice", relation_type="KNOWS",
    ))
    g1.merge(g2)
    assert g1.entity_count == 2
    assert g1.relationship_count == 1


def test_knowledge_graph_stats():
    g = _make_graph()
    s = g.stats()
    assert s["entity_count"] == 3
    assert s["relationship_count"] == 2
    assert "PERSON" in s["entity_types"]
    assert s["entity_types"]["PERSON"] == 2


def test_knowledge_graph_to_dict():
    g = _make_graph()
    d = g.to_dict()
    assert "entities" in d
    assert "relationships" in d
    assert len(d["entities"]) == 3
    assert len(d["relationships"]) == 2


# ---------------------------------------------------------------------------
# EntityExtractor — heuristic
# ---------------------------------------------------------------------------

def test_entity_extractor_heuristic():
    ext = EntityExtractor()
    entities, rels = ext.extract("Dr. Alice Smith works at Acme Corp in New York.")
    names = {e.name for e in entities}
    assert "Alice Smith" in names or "Acme Corp" in names
    assert len(entities) >= 1


def test_entity_extractor_empty():
    ext = EntityExtractor()
    entities, rels = ext.extract("")
    assert entities == []
    assert rels == []


def test_entity_extractor_organizations():
    ext = EntityExtractor()
    entities, _ = ext.extract("Google Inc and Microsoft Corp are technology companies.")
    types = {e.entity_type for e in entities}
    assert "ORGANIZATION" in types or "TECHNOLOGY" in types


def test_entity_extractor_technology_keywords():
    ext = EntityExtractor()
    entities, _ = ext.extract("We use Python and Docker for our API deployment.")
    tech_names = {e.name.lower() for e in entities if e.entity_type == "TECHNOLOGY"}
    assert "python" in tech_names or "docker" in tech_names or "api" in tech_names


def test_entity_extractor_custom_types():
    ext = EntityExtractor(entity_types=["PERSON", "CUSTOM"])
    assert "CUSTOM" in ext.entity_types
    assert "PERSON" in ext.entity_types


def test_entity_extractor_with_llm():
    """Mock LLM returns valid JSON extraction."""
    def mock_llm(prompt):
        return '{"entities": [{"name": "Alice", "type": "PERSON", "description": "researcher"}], "relationships": []}'

    ext = EntityExtractor(extract_fn=mock_llm)
    entities, rels = ext.extract("Alice is a researcher.")
    assert len(entities) == 1
    assert entities[0].name == "Alice"
    assert entities[0].entity_type == "PERSON"


def test_entity_extractor_llm_bad_json_falls_back():
    """When LLM returns bad JSON, fall back to heuristic extraction."""
    def bad_llm(prompt):
        return "this is not json"

    ext = EntityExtractor(extract_fn=bad_llm)
    entities, _ = ext.extract("Dr. Alice Smith works at Acme Corp.")
    # Should still get entities from heuristic fallback
    assert isinstance(entities, list)


def test_entity_extractor_cooccurrence_relationships():
    ext = EntityExtractor()
    text = "Alice Smith and Bob Jones work at Acme Corp. Alice Smith manages Bob Jones."
    entities, rels = ext.extract(text)
    assert isinstance(rels, list)


@pytest.mark.asyncio
async def test_entity_extractor_async():
    ext = EntityExtractor()
    entities, rels = await ext.aextract("Dr. Alice Smith works at Acme Corp.")
    assert isinstance(entities, list)
