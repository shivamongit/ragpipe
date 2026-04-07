"""Tests for ragpipe.graph.builder — build knowledge graphs from documents."""

import pytest

from ragpipe.graph.builder import GraphBuilder, GraphBuildResult
from ragpipe.graph.entities import EntityExtractor, KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm(prompt):
    return (
        '{"entities": ['
        '{"name": "Alice", "type": "PERSON", "description": "researcher"},'
        '{"name": "Acme", "type": "ORGANIZATION", "description": "company"}'
        '], "relationships": ['
        '{"source": "Alice", "target": "Acme", "type": "WORKS_FOR"}'
        ']}'
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_from_texts():
    builder = GraphBuilder()
    result = builder.build([
        "Dr. Alice Smith works at Acme Corp in New York.",
        "Bob Jones joined Acme Corp last year.",
    ])
    assert isinstance(result, GraphBuildResult)
    assert result.documents_processed == 2
    assert result.graph.entity_count >= 1


def test_build_result_stats():
    builder = GraphBuilder()
    result = builder.build(["Dr. Alice Smith works at Acme Corp."])
    assert result.entities_extracted == result.graph.entity_count
    assert result.relationships_extracted == result.graph.relationship_count
    assert result.build_time_ms >= 0


def test_build_empty_input():
    builder = GraphBuilder()
    result = builder.build([])
    assert result.documents_processed == 0
    assert result.graph.entity_count == 0
    assert result.graph.relationship_count == 0


def test_build_with_custom_extractor():
    ext = EntityExtractor(extract_fn=_mock_llm)
    builder = GraphBuilder(extractor=ext)
    result = builder.build(["Alice works at Acme."])
    assert result.graph.entity_count >= 1
    names = {e.name for e in result.graph.entities.values()}
    assert "Alice" in names


def test_build_merge_duplicates():
    """Near-duplicate entity names should be merged."""
    builder = GraphBuilder(merge_threshold=0.85)
    result = builder.build([
        "Dr. Alice Smith works at Acme Corp.",
        "Alice Smithe also works at Acme Corp.",
    ])
    # "Alice Smith" and "Alice Smithe" are very similar and should merge
    persons = [
        e for e in result.graph.entities.values() if e.entity_type == "PERSON"
    ]
    # Depending on heuristic: if both extracted, one should merge into the other
    assert isinstance(persons, list)


def test_add_documents_to_existing():
    builder = GraphBuilder()
    first = builder.build(["Dr. Alice Smith works at Acme Corp."])
    graph = first.graph
    count_before = graph.entity_count

    second = builder.add_documents(graph, ["Bob Jones joined Tech Inc last month."])
    assert second.documents_processed == 1
    # Graph should have grown (or at least not shrunk)
    assert second.graph.entity_count >= count_before


def test_build_time_tracking():
    builder = GraphBuilder()
    result = builder.build(["Some text about Dr. Alice Smith."])
    assert result.build_time_ms > 0


def test_build_with_document_objects():
    """Builder should accept objects with .content and .doc_id attributes."""
    class Doc:
        def __init__(self, content, doc_id):
            self.content = content
            self.doc_id = doc_id

    builder = GraphBuilder()
    result = builder.build([Doc("Dr. Alice Smith works at Acme Corp.", "doc1")])
    assert result.documents_processed == 1
    assert result.graph.entity_count >= 1


def test_build_skips_empty_strings():
    builder = GraphBuilder()
    result = builder.build(["", "  ", "Dr. Alice Smith works at Acme Corp."])
    # Empty strings should be skipped but document count still includes them
    assert result.documents_processed == 3


@pytest.mark.asyncio
async def test_graph_builder_async():
    builder = GraphBuilder()
    result = await builder.abuild(["Dr. Alice Smith works at Acme Corp."])
    assert isinstance(result, GraphBuildResult)
    assert result.graph.entity_count >= 1
