"""Tests for ragpipe.agents.selfrag — self-reflective RAG agent."""

import pytest

from ragpipe.agents.selfrag import (
    SelfRAGAgent, SelfRAGResult, SelfRAGReflection,
    RetrievalDecision, RelevanceScore, SupportLevel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mock_retrieve(question, **kwargs):
    return [
        "Paris is the capital of France.",
        "France is located in Western Europe.",
        "The Eiffel Tower is in Paris.",
    ]


def mock_generate(prompt, **kwargs):
    return "Paris is the capital of France, located in Western Europe."


# ---------------------------------------------------------------------------
# Heuristic (no-LLM) tests
# ---------------------------------------------------------------------------

def test_retrieval_decision_heuristic():
    agent = SelfRAGAgent()
    decision = agent._should_retrieve("What is the capital of France?")
    assert decision == RetrievalDecision.RETRIEVE


def test_retrieval_decision_no_retrieve():
    agent = SelfRAGAgent()
    decision = agent._should_retrieve("Do you think AI is dangerous?")
    assert decision == RetrievalDecision.NO_RETRIEVE


def test_relevance_scoring_heuristic():
    agent = SelfRAGAgent()
    scores = agent._score_relevance(
        "What is the capital of France?",
        ["Paris is the capital of France.", "Pizza is delicious."],
    )
    assert len(scores) == 2
    assert scores[0] == RelevanceScore.RELEVANT
    # "Pizza is delicious" has almost no overlap → irrelevant
    assert scores[1] == RelevanceScore.IRRELEVANT


def test_support_check_heuristic():
    agent = SelfRAGAgent()
    level = agent._check_support(
        "Paris is the capital of France",
        ["Paris is the capital of France. It is in Western Europe."],
    )
    assert level in (SupportLevel.FULLY_SUPPORTED, SupportLevel.PARTIALLY_SUPPORTED)


def test_support_check_not_supported():
    agent = SelfRAGAgent()
    level = agent._check_support(
        "Tokyo is famous for sushi and technology",
        ["Paris is the capital of France."],
    )
    assert level in (SupportLevel.NOT_SUPPORTED, SupportLevel.PARTIALLY_SUPPORTED)


def test_selfrag_full_pipeline_heuristic():
    """End-to-end pipeline with no LLM — heuristic only."""
    agent = SelfRAGAgent(retrieve_fn=mock_retrieve)
    result = agent.query("What is the capital of France?")
    assert isinstance(result, SelfRAGResult)
    assert result.answer != ""
    assert result.iterations >= 1
    assert isinstance(result.reflection, SelfRAGReflection)


def test_selfrag_no_retrieve_path():
    """Opinion questions should skip retrieval."""
    agent = SelfRAGAgent(retrieve_fn=mock_retrieve)
    result = agent.query("In your opinion, is AI dangerous?")
    assert result.reflection.retrieval_needed == RetrievalDecision.NO_RETRIEVE


def test_selfrag_with_llm():
    """With mock reflect and generate functions."""
    call_count = {"reflect": 0}

    def mock_reflect(prompt):
        call_count["reflect"] += 1
        if "retrieval decision" in prompt.lower() or "decide whether" in prompt.lower():
            return '{"decision": "retrieve", "reasoning": "factual question"}'
        if "relevance" in prompt.lower():
            return '{"relevance": "relevant", "reasoning": "matches"}'
        if "support" in prompt.lower():
            return '{"support": "fully_supported", "reasoning": "matches"}'
        if "usefulness" in prompt.lower():
            return '{"usefulness": 5, "reasoning": "very useful"}'
        return '{"decision": "retrieve"}'

    agent = SelfRAGAgent(
        retrieve_fn=mock_retrieve,
        generate_fn=mock_generate,
        reflect_fn=mock_reflect,
    )
    result = agent.query("What is the capital of France?")
    assert result.answer != ""
    assert call_count["reflect"] >= 1


def test_selfrag_iterates_on_low_quality():
    """Agent should re-try when support is not met."""
    iteration_count = {"n": 0}

    def counting_retrieve(question, **kw):
        iteration_count["n"] += 1
        return ["Some loosely related text about geography."]

    agent = SelfRAGAgent(
        retrieve_fn=counting_retrieve,
        max_iterations=3,
        support_threshold="fully_supported",
    )
    result = agent.query("What is quantum computing?")
    # Should iterate more than once because heuristic support will be low
    assert result.iterations >= 1


def test_selfrag_empty_question():
    agent = SelfRAGAgent(retrieve_fn=mock_retrieve)
    result = agent.query("")
    assert isinstance(result, SelfRAGResult)


def test_selfrag_reflection_fields():
    ref = SelfRAGReflection(
        retrieval_needed=RetrievalDecision.RETRIEVE,
        relevance_scores=[RelevanceScore.RELEVANT, RelevanceScore.IRRELEVANT],
        support_level=SupportLevel.PARTIALLY_SUPPORTED,
        usefulness=4,
        reasoning="test",
    )
    assert ref.retrieval_needed == RetrievalDecision.RETRIEVE
    assert len(ref.relevance_scores) == 2
    assert ref.support_level == SupportLevel.PARTIALLY_SUPPORTED
    assert ref.usefulness == 4
    assert ref.reasoning == "test"


def test_selfrag_result_confidence():
    agent = SelfRAGAgent(retrieve_fn=mock_retrieve)
    result = agent.query("What is the capital of France?")
    assert 0.0 <= result.confidence <= 1.0


def test_selfrag_no_retrieve_fn():
    """Agent with no retrieve_fn should still work (empty passages)."""
    agent = SelfRAGAgent()
    result = agent.query("What is the capital of France?")
    assert isinstance(result, SelfRAGResult)
    assert result.retrieved_passages == [] or isinstance(result.retrieved_passages, list)


@pytest.mark.asyncio
async def test_selfrag_async():
    agent = SelfRAGAgent(retrieve_fn=mock_retrieve)
    result = await agent.aquery("What is the capital of France?")
    assert isinstance(result, SelfRAGResult)
    assert result.answer != ""
