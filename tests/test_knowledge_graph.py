"""Tests for ragpipe.graph — Knowledge Graph RAG."""

from ragpipe.graph.knowledge_graph import (
    KnowledgeGraph, Entity, Relation, Triple, GraphSearchResult,
    _parse_triples, _heuristic_extract,
)


# ── Triple ────────────────────────────────────────────────────────────────────

def test_triple_to_dict():
    t = Triple(subject="Paris", predicate="capital_of", object="France")
    d = t.to_dict()
    assert d["subject"] == "Paris"
    assert d["predicate"] == "capital_of"
    assert d["object"] == "France"


# ── _parse_triples ────────────────────────────────────────────────────────────

def test_parse_triples_json():
    raw = '[{"subject": "A", "predicate": "rel", "object": "B"}]'
    triples = _parse_triples(raw)
    assert len(triples) == 1
    assert triples[0].subject == "A"


def test_parse_triples_embedded_json():
    raw = 'Here are the triples:\n[{"subject": "X", "predicate": "is", "object": "Y"}]\nDone.'
    triples = _parse_triples(raw)
    assert len(triples) == 1


def test_parse_triples_invalid():
    triples = _parse_triples("not valid json at all")
    assert triples == []


def test_parse_triples_skips_empty():
    raw = '[{"subject": "", "predicate": "rel", "object": "B"}, {"subject": "A", "predicate": "rel", "object": "B"}]'
    triples = _parse_triples(raw)
    assert len(triples) == 1


# ── _heuristic_extract ────────────────────────────────────────────────────────

def test_heuristic_extract_is_pattern():
    triples = _heuristic_extract("Paris is the capital of France.")
    assert len(triples) >= 1
    assert any("Paris" in t.subject for t in triples)


def test_heuristic_extract_located_in():
    triples = _heuristic_extract("France located in Europe.")
    assert len(triples) >= 1


def test_heuristic_extract_empty():
    triples = _heuristic_extract("")
    assert triples == []


# ── KnowledgeGraph construction ───────────────────────────────────────────────

def test_kg_add_triple():
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="Paris", predicate="capital_of", object="France"))
    assert kg.entity_count == 2
    assert kg.triple_count == 1


def test_kg_add_triples():
    kg = KnowledgeGraph()
    kg.add_triples([
        Triple(subject="Paris", predicate="capital_of", object="France"),
        Triple(subject="France", predicate="located_in", object="Europe"),
    ])
    assert kg.entity_count == 3
    assert kg.triple_count == 2


def test_kg_entity_mentions():
    kg = KnowledgeGraph()
    kg.add_triples([
        Triple(subject="Paris", predicate="capital_of", object="France"),
        Triple(subject="France", predicate="located_in", object="Europe"),
    ])
    france = kg.get_entity("France")
    assert france is not None
    assert france.mentions == 2


def test_kg_add_document_heuristic():
    kg = KnowledgeGraph()
    triples = kg.add_document("Paris is the capital of France. France is a country.")
    assert len(triples) >= 1
    assert kg.entity_count >= 1


def test_kg_add_document_with_llm():
    def mock_extract(prompt):
        return '[{"subject": "AI", "predicate": "field_of", "object": "Computer Science"}]'

    kg = KnowledgeGraph(extract_fn=mock_extract)
    triples = kg.add_document("AI is a field of computer science.")
    assert len(triples) == 1
    assert kg.get_entity("ai") is not None


# ── Graph search ──────────────────────────────────────────────────────────────

def test_kg_search_basic():
    kg = KnowledgeGraph()
    kg.add_triples([
        Triple(subject="Paris", predicate="capital_of", object="France"),
        Triple(subject="France", predicate="located_in", object="Europe"),
    ])
    results = kg.search("Paris")
    assert len(results) >= 1
    assert results[0].entity == "Paris"


def test_kg_search_multi_hop():
    kg = KnowledgeGraph()
    kg.add_triples([
        Triple(subject="Paris", predicate="capital_of", object="France"),
        Triple(subject="France", predicate="located_in", object="Europe"),
        Triple(subject="Europe", predicate="part_of", object="Earth"),
    ])
    results = kg.search("Paris", max_hops=3)
    assert len(results) >= 1
    # Should find entities beyond direct neighbors
    related = results[0].related_entities
    assert len(related) >= 1


def test_kg_search_no_match():
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="A", predicate="rel", object="B"))
    results = kg.search("ZZZ_nonexistent")
    assert len(results) == 0


def test_kg_get_neighbors():
    kg = KnowledgeGraph()
    kg.add_triples([
        Triple(subject="Paris", predicate="capital_of", object="France"),
        Triple(subject="Paris", predicate="located_in", object="Ile-de-France"),
    ])
    neighbors = kg.get_neighbors("Paris")
    assert len(neighbors) == 2


# ── Fusion ────────────────────────────────────────────────────────────────────

def test_kg_fuse_empty():
    kg = KnowledgeGraph()
    fused = kg.fuse([], [])
    assert fused == []


def test_kg_fuse_graph_only():
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="A", predicate="rel", object="B"))
    graph_results = kg.search("A")
    fused = kg.fuse(graph_results, [])
    assert len(fused) >= 1
    assert fused[0]["combined_score"] > 0


# ── Serialization ─────────────────────────────────────────────────────────────

def test_kg_to_dict():
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="A", predicate="rel", object="B"))
    d = kg.to_dict()
    assert d["entity_count"] == 2
    assert d["triple_count"] == 1


def test_kg_summary():
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="A", predicate="rel", object="B"))
    s = kg.summary()
    assert "KnowledgeGraph" in s


def test_graph_search_result_to_dict():
    result = GraphSearchResult(
        entity="Test", related_entities=[], paths=[], triples=[], score=0.5,
    )
    d = result.to_dict()
    assert d["entity"] == "Test"
    assert d["score"] == 0.5


def test_kg_clear():
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="A", predicate="rel", object="B"))
    kg.clear()
    assert kg.entity_count == 0
    assert kg.triple_count == 0
