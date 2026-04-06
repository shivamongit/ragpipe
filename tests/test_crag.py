"""Tests for Corrective RAG (CRAG) agent."""

from ragpipe.agents.crag import (
    CRAGAgent, CRAGAction, CRAGResult, RelevanceGrade,
    GradedDocument, _parse_grade,
)


def test_parse_grade_correct():
    raw = '{"grade": "correct", "confidence": 0.9, "reasoning": "relevant"}'
    grade, conf, reason = _parse_grade(raw)
    assert grade == RelevanceGrade.CORRECT
    assert conf == 0.9


def test_parse_grade_incorrect():
    raw = '{"grade": "incorrect", "confidence": 0.2, "reasoning": "off topic"}'
    grade, conf, reason = _parse_grade(raw)
    assert grade == RelevanceGrade.INCORRECT
    assert conf == 0.2


def test_parse_grade_ambiguous():
    raw = '{"grade": "ambiguous", "confidence": 0.5, "reasoning": "partial"}'
    grade, conf, _ = _parse_grade(raw)
    assert grade == RelevanceGrade.AMBIGUOUS


def test_parse_grade_fallback_keyword():
    grade, _, _ = _parse_grade("The document is correct and relevant")
    assert grade == RelevanceGrade.CORRECT


def test_parse_grade_fallback_incorrect():
    grade, _, _ = _parse_grade("totally unrelated garbage")
    assert grade == RelevanceGrade.INCORRECT


def test_crag_no_retrieve_fn():
    agent = CRAGAgent()
    result = agent.query("What is X?")
    assert result.action_taken == CRAGAction.NO_ANSWER
    assert result.confidence == 0.0


def test_crag_direct_generate():
    """When all docs are graded CORRECT, generate directly."""
    def grade_fn(prompt):
        return '{"grade": "correct", "confidence": 0.95, "reasoning": "relevant"}'

    def retrieve_fn(q, **kw):
        return ["Paris is the capital of France.", "France is in Europe."]

    def generate_fn(prompt):
        return "Paris is the capital of France."

    agent = CRAGAgent(
        grade_fn=grade_fn,
        retrieve_fn=retrieve_fn,
        generate_fn=generate_fn,
    )
    result = agent.query("What is the capital of France?")
    assert result.action_taken == CRAGAction.DIRECT_GENERATE
    assert result.confidence > 0.8
    assert "Paris" in result.answer
    assert result.sources_used == 2


def test_crag_refined_generate():
    """When docs are ambiguous, refine knowledge then generate."""
    call_count = {"n": 0}

    def grade_fn(prompt):
        call_count["n"] += 1
        if call_count["n"] <= 1:
            return '{"grade": "ambiguous", "confidence": 0.5, "reasoning": "partial"}'
        return '{"grade": "ambiguous", "confidence": 0.4, "reasoning": "tangential"}'

    def retrieve_fn(q, **kw):
        return ["Some tangential info about France.", "Also some tourism data."]

    def generate_fn(prompt):
        return "Based on available information, France is relevant."

    agent = CRAGAgent(
        grade_fn=grade_fn,
        retrieve_fn=retrieve_fn,
        generate_fn=generate_fn,
        relevance_threshold=0.3,
    )
    result = agent.query("Tell me about France")
    assert result.action_taken == CRAGAction.REFINED_GENERATE
    assert len(result.graded_docs) == 2


def test_crag_web_search_fallback():
    """When all docs are INCORRECT and web_search_fn exists, fall back to web."""
    def grade_fn(prompt):
        return '{"grade": "incorrect", "confidence": 0.1, "reasoning": "irrelevant"}'

    def retrieve_fn(q, **kw):
        return ["Completely unrelated document about cooking."]

    def web_search_fn(q):
        return ["Web result: Paris is the capital of France."]

    def generate_fn(prompt):
        return "Paris is the capital of France."

    agent = CRAGAgent(
        grade_fn=grade_fn,
        retrieve_fn=retrieve_fn,
        generate_fn=generate_fn,
        web_search_fn=web_search_fn,
    )
    result = agent.query("What is the capital of France?")
    assert result.action_taken == CRAGAction.WEB_SEARCH
    assert result.metadata.get("web_search") is True


def test_crag_no_answer():
    """When docs are INCORRECT and no web search, return no_answer."""
    def grade_fn(prompt):
        return '{"grade": "incorrect", "confidence": 0.1, "reasoning": "irrelevant"}'

    def retrieve_fn(q, **kw):
        return ["Unrelated document."]

    agent = CRAGAgent(grade_fn=grade_fn, retrieve_fn=retrieve_fn)
    result = agent.query("What is quantum computing?")
    assert result.action_taken == CRAGAction.NO_ANSWER
    assert result.confidence == 0.0
    assert "don't have enough" in result.answer.lower() or "i don" in result.answer.lower()


def test_crag_no_grade_fn():
    """Without a grade function, all docs assumed CORRECT."""
    def retrieve_fn(q, **kw):
        return ["Document about topic."]

    def generate_fn(prompt):
        return "Answer from docs."

    agent = CRAGAgent(retrieve_fn=retrieve_fn, generate_fn=generate_fn)
    result = agent.query("Question?")
    assert result.action_taken == CRAGAction.DIRECT_GENERATE
    assert result.sources_used == 1


def test_crag_empty_docs():
    def retrieve_fn(q, **kw):
        return []

    agent = CRAGAgent(retrieve_fn=retrieve_fn)
    result = agent.query("Question?")
    assert result.action_taken == CRAGAction.NO_ANSWER
