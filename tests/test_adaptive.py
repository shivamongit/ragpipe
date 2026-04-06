"""Tests for Adaptive Retrieval agent."""

from ragpipe.agents.adaptive import (
    AdaptiveRetriever, AdaptiveResult, QueryComplexity,
    RetrievalStrategy, _parse_complexity, STRATEGY_MAP,
)


def test_parse_complexity_factual():
    raw = '{"complexity": "factual", "reasoning": "simple lookup"}'
    assert _parse_complexity(raw) == QueryComplexity.FACTUAL


def test_parse_complexity_comparative():
    raw = '{"complexity": "comparative", "reasoning": "compares two items"}'
    assert _parse_complexity(raw) == QueryComplexity.COMPARATIVE


def test_parse_complexity_analytical():
    raw = '{"complexity": "analytical", "reasoning": "needs explanation"}'
    assert _parse_complexity(raw) == QueryComplexity.ANALYTICAL


def test_parse_complexity_exploratory():
    raw = '{"complexity": "exploratory", "reasoning": "broad overview"}'
    assert _parse_complexity(raw) == QueryComplexity.EXPLORATORY


def test_parse_complexity_fallback_keyword():
    assert _parse_complexity("this is a comparison") == QueryComplexity.COMPARATIVE
    assert _parse_complexity("give me an overview") == QueryComplexity.EXPLORATORY


def test_heuristic_classify_factual():
    c = AdaptiveRetriever._heuristic_classify("What is FAISS?")
    assert c == QueryComplexity.FACTUAL


def test_heuristic_classify_comparative():
    c = AdaptiveRetriever._heuristic_classify("Compare FAISS vs ChromaDB")
    assert c == QueryComplexity.COMPARATIVE


def test_heuristic_classify_analytical():
    c = AdaptiveRetriever._heuristic_classify("Why does hybrid search work better?")
    assert c == QueryComplexity.ANALYTICAL


def test_heuristic_classify_exploratory():
    c = AdaptiveRetriever._heuristic_classify("Give me an overview of vector databases")
    assert c == QueryComplexity.EXPLORATORY


def test_heuristic_classify_conversational():
    c = AdaptiveRetriever._heuristic_classify("What about the second point?")
    assert c == QueryComplexity.CONVERSATIONAL


def test_strategy_map_coverage():
    """Every complexity has a strategy mapping."""
    for complexity in QueryComplexity:
        assert complexity in STRATEGY_MAP


def test_adaptive_no_strategies():
    retriever = AdaptiveRetriever()
    result = retriever.retrieve("test query")
    assert result.documents == []
    assert result.confidence == 0.0


def test_adaptive_single_strategy():
    def dense_fn(q, top_k=5):
        return [f"result_{i}" for i in range(top_k)]

    retriever = AdaptiveRetriever(strategies={"dense": dense_fn})
    result = retriever.retrieve("What is FAISS?")
    assert len(result.documents) > 0
    assert result.query_complexity == QueryComplexity.FACTUAL
    assert result.confidence > 0


def test_adaptive_comparative_uses_more_docs():
    def dense_fn(q, top_k=5):
        return [f"dense_{i}" for i in range(top_k)]

    def sparse_fn(q, top_k=5):
        return [f"sparse_{i}" for i in range(top_k)]

    retriever = AdaptiveRetriever(
        strategies={"dense": dense_fn, "sparse": sparse_fn},
    )
    result = retriever.retrieve("Compare FAISS vs ChromaDB performance")
    assert result.query_complexity == QueryComplexity.COMPARATIVE
    assert result.top_k_used >= 5  # Comparative queries get higher top_k


def test_adaptive_with_classify_fn():
    def classify_fn(prompt):
        return '{"complexity": "analytical", "reasoning": "needs reasoning"}'

    def dense_fn(q, top_k=5):
        return ["doc1", "doc2"]

    retriever = AdaptiveRetriever(
        classify_fn=classify_fn,
        strategies={"dense": dense_fn, "hybrid": dense_fn},
    )
    result = retriever.retrieve("Why is hybrid search better?")
    assert result.query_complexity == QueryComplexity.ANALYTICAL


def test_adaptive_confidence_scoring():
    """Documents with score attributes should improve confidence."""
    class ScoredDoc:
        def __init__(self, text, score):
            self.text = text
            self.score = score

    def dense_fn(q, top_k=5):
        return [ScoredDoc(f"doc_{i}", 0.9) for i in range(3)]

    retriever = AdaptiveRetriever(strategies={"dense": dense_fn})
    result = retriever.retrieve("What is X?")
    assert result.confidence > 0.5


def test_adaptive_custom_top_k():
    def dense_fn(q, top_k=5):
        return [f"doc_{i}" for i in range(top_k)]

    retriever = AdaptiveRetriever(strategies={"dense": dense_fn})
    result = retriever.retrieve("What is X?", top_k=7)
    assert result.top_k_used == 7
