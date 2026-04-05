"""Tests for LLM-as-Judge evaluation."""

import json
import pytest

from ragpipe.evaluation.llm_judge import LLMJudge, JudgmentResult, _parse_judgment


def test_parse_judgment_valid():
    resp = '{"score": 4, "reasoning": "well grounded"}'
    result = _parse_judgment(resp, "faithfulness")
    assert result.score == 4.0
    assert result.reasoning == "well grounded"
    assert result.dimension == "faithfulness"


def test_parse_judgment_clamp():
    resp = '{"score": 10, "reasoning": "too high"}'
    result = _parse_judgment(resp, "relevance")
    assert result.score == 5.0

    resp = '{"score": -2, "reasoning": "too low"}'
    result = _parse_judgment(resp, "relevance")
    assert result.score == 0.0


def test_parse_judgment_garbage():
    result = _parse_judgment("no json here", "completeness")
    assert result.score == 0.0
    assert "Failed" in result.reasoning


def test_parse_judgment_embedded_json():
    resp = 'Here is my evaluation:\n{"score": 3, "reasoning": "partial"}\nDone.'
    result = _parse_judgment(resp, "faithfulness")
    assert result.score == 3.0


def test_judge_no_fn():
    judge = LLMJudge()
    scores = judge.evaluate("What?", "Answer.", ["Context."])
    assert scores["faithfulness"].score == 0
    assert scores["overall"] == 0.0


def test_judge_evaluate_with_mock():
    def mock_judge(prompt):
        if "faithfulness" in prompt.lower():
            return '{"score": 5, "reasoning": "fully supported"}'
        elif "relevance" in prompt.lower():
            return '{"score": 4, "reasoning": "addresses question"}'
        elif "completeness" in prompt.lower():
            return '{"score": 3, "reasoning": "misses some details"}'
        return '{"score": 0, "reasoning": "unknown"}'

    judge = LLMJudge(judge_fn=mock_judge)
    scores = judge.evaluate(
        question="What is FAISS?",
        answer="FAISS is a library for vector search.",
        context_texts=["FAISS is developed by Meta for similarity search."],
    )

    assert scores["faithfulness"].score == 5.0
    assert scores["relevance"].score == 4.0
    assert scores["completeness"].score == 3.0
    # 5*0.4 + 4*0.35 + 3*0.25 = 2.0 + 1.4 + 0.75 = 4.15
    assert scores["overall"] == 4.15


def test_judge_individual_dimensions():
    def mock_judge(prompt):
        return '{"score": 4, "reasoning": "good"}'

    judge = LLMJudge(judge_fn=mock_judge)

    f = judge.judge_faithfulness("q", "a", ["ctx"])
    assert f.dimension == "faithfulness"
    assert f.score == 4.0

    r = judge.judge_relevance("q", "a")
    assert r.dimension == "relevance"

    c = judge.judge_completeness("q", "a", ["ctx"])
    assert c.dimension == "completeness"


@pytest.mark.asyncio
async def test_judge_aevaluate():
    async def amock_judge(prompt):
        return '{"score": 5, "reasoning": "perfect"}'

    judge = LLMJudge(ajudge_fn=amock_judge)
    scores = await judge.aevaluate("q", "a", ["ctx"])
    assert scores["faithfulness"].score == 5.0
    assert scores["relevance"].score == 5.0
    assert scores["completeness"].score == 5.0
    assert scores["overall"] == 5.0


def test_judge_custom_weights():
    def mock_judge(prompt):
        return '{"score": 4, "reasoning": "ok"}'

    judge = LLMJudge(
        judge_fn=mock_judge,
        weights={"faithfulness": 1.0, "relevance": 0.0, "completeness": 0.0},
    )
    scores = judge.evaluate("q", "a", ["ctx"])
    assert scores["overall"] == 4.0
