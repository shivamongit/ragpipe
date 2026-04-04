"""Tests for query expansion strategies."""

from ragpipe.query.expansion import HyDEExpander, MultiQueryExpander, StepBackExpander


def _fake_llm(prompt: str) -> str:
    """Deterministic fake LLM for testing."""
    if "factual paragraph" in prompt.lower():
        return "FAISS is a library developed by Meta for efficient similarity search over dense vectors."
    elif "different search queries" in prompt.lower():
        return "vector similarity search library\nefficient nearest neighbor search\ndense embedding retrieval"
    elif "more general question" in prompt.lower():
        return "What are the main libraries used for vector search?"
    return "Default response."


def test_hyde_expander():
    expander = HyDEExpander(generate_fn=_fake_llm)
    queries = expander.expand("What is FAISS?")
    assert len(queries) == 2
    assert queries[0] == "What is FAISS?"
    assert "FAISS" in queries[1]


def test_multi_query_expander():
    expander = MultiQueryExpander(generate_fn=_fake_llm, n_queries=3)
    queries = expander.expand("What is FAISS?")
    assert len(queries) >= 2
    assert queries[0] == "What is FAISS?"


def test_step_back_expander():
    expander = StepBackExpander(generate_fn=_fake_llm)
    queries = expander.expand("How does FAISS IndexFlatIP work?")
    assert len(queries) == 2
    assert queries[0] == "How does FAISS IndexFlatIP work?"
    assert "vector search" in queries[1].lower()
