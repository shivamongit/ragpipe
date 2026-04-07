"""Tests for ragpipe.graph.community — community detection in knowledge graphs."""

import pytest

from ragpipe.graph.community import CommunityDetector, Community
from ragpipe.graph.entities import Entity, Relationship, KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_cluster_graph():
    """Two clusters: {Alice, Bob, Acme} and {Charlie, Dave, BigCo}, connected loosely."""
    g = KnowledgeGraph()
    # Cluster 1
    g.add_entity(Entity(name="Alice", entity_type="PERSON", description="engineer"))
    g.add_entity(Entity(name="Bob", entity_type="PERSON", description="designer"))
    g.add_entity(Entity(name="Acme", entity_type="ORGANIZATION"))
    g.add_relationship(Relationship(source="PERSON:alice", target="ORGANIZATION:acme", relation_type="WORKS_FOR", weight=1.0))
    g.add_relationship(Relationship(source="PERSON:bob", target="ORGANIZATION:acme", relation_type="WORKS_FOR", weight=1.0))
    g.add_relationship(Relationship(source="PERSON:alice", target="PERSON:bob", relation_type="KNOWS", weight=1.0))
    # Cluster 2
    g.add_entity(Entity(name="Charlie", entity_type="PERSON"))
    g.add_entity(Entity(name="Dave", entity_type="PERSON"))
    g.add_entity(Entity(name="BigCo", entity_type="ORGANIZATION"))
    g.add_relationship(Relationship(source="PERSON:charlie", target="ORGANIZATION:bigco", relation_type="WORKS_FOR", weight=1.0))
    g.add_relationship(Relationship(source="PERSON:dave", target="ORGANIZATION:bigco", relation_type="WORKS_FOR", weight=1.0))
    g.add_relationship(Relationship(source="PERSON:charlie", target="PERSON:dave", relation_type="KNOWS", weight=1.0))
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_community_detection_basic():
    g = _make_two_cluster_graph()
    detector = CommunityDetector(min_community_size=2)
    communities = detector.detect(g)
    assert len(communities) >= 1
    total = sum(c.size for c in communities)
    assert total == g.entity_count


def test_community_detection_disconnected():
    """Two fully disconnected components should form separate communities."""
    g = KnowledgeGraph()
    g.add_entity(Entity(name="A", entity_type="CONCEPT"))
    g.add_entity(Entity(name="B", entity_type="CONCEPT"))
    g.add_relationship(Relationship(source="CONCEPT:a", target="CONCEPT:b", relation_type="RELATED_TO"))
    g.add_entity(Entity(name="X", entity_type="CONCEPT"))
    g.add_entity(Entity(name="Y", entity_type="CONCEPT"))
    g.add_relationship(Relationship(source="CONCEPT:x", target="CONCEPT:y", relation_type="RELATED_TO"))
    detector = CommunityDetector(min_community_size=2)
    communities = detector.detect(g)
    assert len(communities) >= 2


def test_community_min_size():
    """Communities below min_community_size should be merged."""
    g = _make_two_cluster_graph()
    # Add an isolated node
    g.add_entity(Entity(name="Loner", entity_type="PERSON"))
    detector = CommunityDetector(min_community_size=2)
    communities = detector.detect(g)
    # Loner should be merged into another community
    for c in communities:
        assert c.size >= 2


def test_community_summarize():
    g = _make_two_cluster_graph()

    def mock_summarize(prompt):
        return "A group of people and organisations working together."

    detector = CommunityDetector(summarize_fn=mock_summarize, min_community_size=2)
    communities = detector.detect(g)
    for c in communities:
        assert c.summary != ""


def test_community_empty_graph():
    g = KnowledgeGraph()
    detector = CommunityDetector()
    communities = detector.detect(g)
    assert communities == []


def test_community_single_node():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Solo", entity_type="CONCEPT"))
    detector = CommunityDetector(min_community_size=1)
    communities = detector.detect(g)
    assert len(communities) == 1
    assert communities[0].size == 1


def test_community_to_dict():
    c = Community(community_id=0, entities=["PERSON:alice", "PERSON:bob"], summary="A team")
    assert c.community_id == 0
    assert c.size == 2
    assert c.summary == "A team"


def test_community_summarize_all():
    g = _make_two_cluster_graph()
    detector = CommunityDetector(min_community_size=2)
    communities = detector.detect(g)
    # Overwrite summaries
    for c in communities:
        c.summary = ""
    communities = detector.summarize_all(g, communities)
    for c in communities:
        assert c.summary != ""


@pytest.mark.asyncio
async def test_community_detection_async():
    g = _make_two_cluster_graph()
    detector = CommunityDetector(min_community_size=2)
    communities = await detector.adetect(g)
    assert len(communities) >= 1
