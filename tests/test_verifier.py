"""Tests for Answer Verifier — hallucination detection and citation grounding."""

from ragpipe.verification.verifier import (
    AnswerVerifier, VerificationResult, ClaimVerification,
    _parse_claims, _parse_verification, _simple_claim_split, _simple_verify,
)


def test_parse_claims_json():
    raw = '["Paris is the capital.", "It has 2M people."]'
    claims = _parse_claims(raw)
    assert len(claims) == 2
    assert "Paris" in claims[0]


def test_parse_claims_fallback():
    raw = "1. First claim here.\n2. Second claim here."
    claims = _parse_claims(raw)
    assert len(claims) >= 2


def test_parse_verification_supported():
    raw = '{"supported": true, "confidence": 0.95, "supporting_source": "Paris is capital", "reasoning": "direct match"}'
    supported, conf, source, reason = _parse_verification(raw)
    assert supported is True
    assert conf == 0.95
    assert "Paris" in source


def test_parse_verification_unsupported():
    raw = '{"supported": false, "confidence": 0.2, "supporting_source": "", "reasoning": "not found"}'
    supported, conf, source, reason = _parse_verification(raw)
    assert supported is False
    assert conf == 0.2


def test_parse_verification_fallback():
    supported, conf, source, reason = _parse_verification("This claim is not supported at all")
    assert supported is False


def test_simple_claim_split():
    text = "Paris is the capital of France. It has 2.1 million residents. The Eiffel Tower is there."
    claims = _simple_claim_split(text)
    assert len(claims) == 3


def test_simple_verify_supported():
    supported, score, source = _simple_verify(
        "Paris is the capital of France",
        ["Paris is the capital of France and a major European city."],
    )
    assert supported is True
    assert score > 0.5


def test_simple_verify_unsupported():
    supported, score, source = _simple_verify(
        "Tokyo is the largest city in the world",
        ["Paris is the capital of France."],
    )
    assert supported is False
    assert score < 0.5


def test_verifier_heuristic_mode():
    """Verifier works without LLM using word-overlap heuristics."""
    verifier = AnswerVerifier()
    result = verifier.verify(
        answer="Paris is the capital of France. Mars is a red planet.",
        sources=["Paris is the capital of France and a major city."],
    )
    assert result.total_claims == 2
    assert result.supported_claims >= 1
    assert result.hallucination_rate > 0  # Mars claim is unsupported
    assert result.overall_confidence > 0


def test_verifier_all_supported():
    verifier = AnswerVerifier()
    result = verifier.verify(
        answer="Paris is the capital of France.",
        sources=["Paris is the capital of France."],
    )
    assert result.supported_claims >= 1
    assert result.hallucination_rate == 0.0


def test_verifier_empty_answer():
    verifier = AnswerVerifier()
    result = verifier.verify(answer="", sources=["some source"])
    assert result.total_claims == 0
    assert result.hallucination_rate == 0.0


def test_verifier_grounded_answer():
    """Grounded answer should only include supported claims."""
    verifier = AnswerVerifier()
    result = verifier.verify(
        answer="Paris is the capital of France. Aliens live on Jupiter.",
        sources=["Paris is the capital of France and a European city."],
    )
    assert "Paris" in result.grounded_answer
    # Aliens claim should not be in grounded answer
    assert "Aliens" not in result.grounded_answer or result.unsupported_claims > 0


def test_verifier_with_llm():
    """Verifier with mock LLM functions."""
    def decompose_fn(prompt):
        return '["Paris is the capital.", "It has 2M people."]'

    def verify_fn(prompt):
        # The claim text appears in the prompt as "Claim: <text>"
        if "2M people" in prompt:
            return '{"supported": false, "confidence": 0.1, "supporting_source": "", "reasoning": "not found in sources"}'
        return '{"supported": true, "confidence": 0.95, "supporting_source": "Paris is capital", "reasoning": "match"}'

    verifier = AnswerVerifier(verify_fn=verify_fn, decompose_fn=decompose_fn)
    result = verifier.verify(
        answer="Paris is the capital. It has 2M people.",
        sources=["Paris is the capital of France."],
    )
    assert result.total_claims == 2
    assert result.supported_claims == 1
    assert result.unsupported_claims == 1
    assert result.hallucination_rate == 0.5


def test_verification_result_to_dict():
    result = VerificationResult(
        claims=[ClaimVerification(text="test", supported=True, confidence=0.9)],
        overall_confidence=0.9,
        hallucination_rate=0.0,
        grounded_answer="test",
        total_claims=1,
        supported_claims=1,
        unsupported_claims=0,
    )
    d = result.to_dict()
    assert d["overall_confidence"] == 0.9
    assert len(d["claims"]) == 1
    assert d["claims"][0]["supported"] is True
