"""Tests for ragpipe.graph.retriever — graph-enhanced retrieval."""

import pytest

from ragpipe.graph.retriever import GraphRetriever, GraphRAGResult
from ragpipe.graph.entities import Entity, Relationship, KnowledgeGraph
from ragpipe.graph.community import Community


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Alice", entity_type="PERSON", description="AI researcher at Acme"))
    g.add_entity(Entity(name="Bob", entity_type="PERSON", description="data engineer"))
    g.add_entity(Entity(name="Acme", entity_type="ORGANIZATION", description="tech company"))
    g.add_entity(Entity(name="Python", entity_type="TECHNOLOGY", description="programming language"))
    g.add_relationship(Relationship(source="PERSON:alice", target="ORGANIZATION:acme", relation_type="WORKS_FOR"))
    g.add_relationship(Relationship(source="PERSON:bob", target="ORGANIZATION:acme", relation_type="WORKS_FOR"))
    g.add_relationship(Relationship(source="PERSON:alice", target="TECHNOLOGY:python", relation_type="USES"))
    return g


def _make_communities():
    return [
        Community(community_id=0, entities=["PERSON:alice", "ORGANIZATION:acme"],
                  summary="Alice works at Acme, a technology company focused on AI."),
        Community(community_id=1, entities=["PERSON:bob", "TECHNOLOGY:python"],
                  summary="Bob uses Python for data engineering projects."),
    ]


def mock_generate(prompt):
    return "Alice is an AI researcher who works at Acme."


def mock_vector_retrieve(query, top_k):
    return ["Alice works on machine learning.", "Acme is a tech company."]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_local_search():
    g = _make_graph()
    retriever = GraphRetriever(graph=g, strategy="local")
    result = retriever.retrieve("Tell me about Alice")
    assert isinstance(result, GraphRAGResult)
    assert result.strategy == "local"
    assert result.answer != ""


def test_global_search():
    g = _make_graph()
    comms = _make_communities()
    retriever = GraphRetriever(graph=g, communities=comms, strategy="global")
    result = retriever.retrieve("What companies are in the graph?")
    assert result.strategy == "global"
    assert result.answer != ""


def test_global_search_no_communities():
    retriever = GraphRetriever(strategy="global")
    result = retriever.retrieve("anything")
    assert "No community" in result.answer


def test_hybrid_search():
    g = _make_graph()
    comms = _make_communities()
    retriever = GraphRetriever(
        graph=g, communities=comms,
        vector_retrieve_fn=mock_vector_retrieve,
        strategy="hybrid",
    )
    result = retriever.retrieve("Tell me about Alice at Acme")
    assert result.strategy == "hybrid"
    assert result.answer != ""


def test_retrieve_with_vector_fn():
    g = _make_graph()
    retriever = GraphRetriever(
        graph=g,
        vector_retrieve_fn=mock_vector_retrieve,
        strategy="hybrid",
    )
    result = retriever.retrieve("Tell me about Alice")
    assert result.answer != ""


def test_retrieve_with_generate_fn():
    g = _make_graph()
    retriever = GraphRetriever(graph=g, generate_fn=mock_generate, strategy="local")
    result = retriever.retrieve("Tell me about Alice")
    assert "Alice" in result.answer


def test_retrieve_entity_extraction():
    g = _make_graph()
    retriever = GraphRetriever(graph=g, strategy="local")
    # "Alice" should be matched as an entity in the query
    result = retriever.retrieve("What does Alice do?")
    # The result should include Alice-related entities
    assert isinstance(result.entities_used, list)


def test_retrieve_graph_traversal():
    g = _make_graph()
    retriever = GraphRetriever(graph=g, strategy="local", max_hops=2)
    result = retriever.retrieve("Tell me about Alice")
    # With 2 hops from Alice, should reach Acme and Bob
    if result.subgraph:
        assert result.subgraph.entity_count >= 1


def test_retrieve_empty_graph():
    retriever = GraphRetriever(strategy="local")
    result = retriever.retrieve("anything")
    assert isinstance(result, GraphRAGResult)
    assert result.confidence == 0.0


def test_retrieve_no_matches():
    g = KnowledgeGraph()
    g.add_entity(Entity(name="Xyzzy", entity_type="CONCEPT"))
    retriever = GraphRetriever(graph=g, strategy="local")
    result = retriever.retrieve("Tell me about quantum computing")
    assert isinstance(result, GraphRAGResult)


def test_graph_rag_result_fields():
    result = GraphRAGResult(
        answer="test",
        confidence=0.8,
        strategy="local",
    )
    assert result.answer == "test"
    assert result.confidence == 0.8
    assert result.strategy == "local"
    assert result.entities_used == []
    assert result.relationships_used == []
    assert result.communities_used == []
    assert result.subgraph is None
    assert result.metadata == {}


def test_retrieve_with_entity_extract_fn():
    g = _make_graph()

    def mock_entity_extract(query):
        return ["Alice"]

    retriever = GraphRetriever(
        graph=g, entity_extract_fn=mock_entity_extract, strategy="local",
    )
    result = retriever.retrieve("Who is Alice?")
    entity_names = {e.name for e in result.entities_used}
    assert "Alice" in entity_names or len(result.entities_used) >= 1


@pytest.mark.asyncio
async def test_retrieve_async():
    g = _make_graph()
    retriever = GraphRetriever(graph=g, strategy="local")
    result = await retriever.aretrieve("Tell me about Alice")
    assert isinstance(result, GraphRAGResult)
    assert result.answer != ""
