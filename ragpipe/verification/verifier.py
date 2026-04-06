"""Answer Verifier — hallucination detection, citation grounding, confidence scoring.

No other RAG framework provides built-in answer verification that:
1. Decomposes an answer into individual claims
2. Checks each claim against source documents for grounding
3. Detects unsupported claims (potential hallucinations)
4. Computes per-claim and overall confidence scores
5. Returns structured verification with cited sources per claim

This is critical for production RAG — you need to know WHICH parts of an
answer are grounded and which are fabricated, not just an overall score.

Usage:
    from ragpipe.verification import AnswerVerifier

    verifier = AnswerVerifier(verify_fn=my_llm)
    result = verifier.verify(
        answer="Paris is the capital of France. It has 5 million people.",
        sources=["Paris is the capital of France.", "Paris has 2.1 million residents."],
    )
    for claim in result.claims:
        print(claim.text, claim.supported, claim.confidence)
    print(result.overall_confidence, result.hallucination_rate)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    text: str
    supported: bool
    confidence: float
    supporting_source: str = ""
    reasoning: str = ""


@dataclass
class VerificationResult:
    """Full verification result for an answer."""
    claims: list[ClaimVerification]
    overall_confidence: float
    hallucination_rate: float
    grounded_answer: str
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_confidence": self.overall_confidence,
            "hallucination_rate": self.hallucination_rate,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "claims": [
                {
                    "text": c.text,
                    "supported": c.supported,
                    "confidence": c.confidence,
                    "supporting_source": c.supporting_source,
                    "reasoning": c.reasoning,
                }
                for c in self.claims
            ],
        }


DECOMPOSE_PROMPT = """Decompose the following answer into individual factual claims. Each claim should be a single, verifiable statement.

Answer: {answer}

Return a JSON array of strings, each being one claim:
["claim 1", "claim 2", "claim 3"]"""

VERIFY_CLAIM_PROMPT = """You are a fact-checker. Given a claim and source documents, determine if the claim is SUPPORTED by the sources.

Claim: {claim}

Source documents:
{sources}

Respond with ONLY a JSON object:
{{"supported": true|false, "confidence": 0.0-1.0, "supporting_source": "quote from source that supports this or empty string", "reasoning": "brief explanation"}}"""


def _parse_claims(raw: str) -> list[str]:
    """Parse LLM response into list of claims."""
    raw = raw.strip()
    try:
        # Try JSON array
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            claims = json.loads(json_match.group())
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if str(c).strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: split by newlines, numbered lists, bullet points
    lines = re.split(r'\n|(?:^\d+[\.\)]\s*)', raw)
    claims = []
    for line in lines:
        line = line.strip().lstrip("-•* ")
        if len(line) > 10:  # Skip very short fragments
            claims.append(line)
    return claims if claims else [raw.strip()]


def _parse_verification(raw: str) -> tuple[bool, float, str, str]:
    """Parse verification response."""
    raw = raw.strip()
    try:
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            supported = bool(data.get("supported", False))
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            source = str(data.get("supporting_source", ""))
            reasoning = str(data.get("reasoning", ""))
            return supported, confidence, source, reasoning
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Keyword fallback
    lower = raw.lower()
    if "supported" in lower and "not supported" not in lower and "unsupported" not in lower:
        return True, 0.7, "", raw[:200]
    return False, 0.3, "", raw[:200]


def _simple_claim_split(answer: str) -> list[str]:
    """Simple sentence-based claim decomposition without LLM."""
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _simple_verify(claim: str, sources: list[str]) -> tuple[bool, float, str]:
    """Simple word-overlap verification without LLM."""
    claim_words = set(claim.lower().split())
    best_overlap = 0.0
    best_source = ""
    for source in sources:
        source_words = set(source.lower().split())
        if not claim_words:
            continue
        overlap = len(claim_words & source_words) / len(claim_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_source = source

    supported = best_overlap >= 0.5
    return supported, best_overlap, best_source[:200]


class AnswerVerifier:
    """Verify RAG answers for hallucination and grounding.

    Decomposes answers into claims, verifies each against source documents,
    and produces structured confidence scores. Works with or without an LLM —
    falls back to word-overlap heuristics when no verify_fn is provided.
    """

    def __init__(
        self,
        verify_fn: Optional[Callable] = None,
        decompose_fn: Optional[Callable] = None,
        min_claim_length: int = 10,
    ):
        self.verify_fn = verify_fn
        self.decompose_fn = decompose_fn
        self.min_claim_length = min_claim_length

    def _decompose_claims(self, answer: str) -> list[str]:
        """Break answer into individual verifiable claims."""
        if self.decompose_fn:
            raw = self.decompose_fn(DECOMPOSE_PROMPT.format(answer=answer))
            claims = _parse_claims(raw)
            if claims:
                return claims
        return _simple_claim_split(answer)

    def _verify_claim(self, claim: str, sources: list[str]) -> ClaimVerification:
        """Verify a single claim against sources."""
        if self.verify_fn:
            sources_text = "\n---\n".join(s[:500] for s in sources[:10])
            prompt = VERIFY_CLAIM_PROMPT.format(claim=claim, sources=sources_text)
            raw = self.verify_fn(prompt)
            supported, confidence, source, reasoning = _parse_verification(raw)
            return ClaimVerification(
                text=claim,
                supported=supported,
                confidence=confidence,
                supporting_source=source,
                reasoning=reasoning,
            )

        # Heuristic verification
        supported, confidence, best_source = _simple_verify(claim, sources)
        return ClaimVerification(
            text=claim,
            supported=supported,
            confidence=confidence,
            supporting_source=best_source,
            reasoning="word-overlap heuristic",
        )

    def verify(self, answer: str, sources: list[str]) -> VerificationResult:
        """Verify an answer against source documents.

        Args:
            answer: The generated answer to verify
            sources: List of source document texts

        Returns:
            VerificationResult with per-claim and overall verification
        """
        if not answer or not answer.strip():
            return VerificationResult(
                claims=[],
                overall_confidence=0.0,
                hallucination_rate=0.0,
                grounded_answer="",
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
            )

        # Step 1: Decompose into claims
        claims = self._decompose_claims(answer)

        # Step 2: Verify each claim
        verified = [self._verify_claim(claim, sources) for claim in claims]

        # Step 3: Compute metrics
        total = len(verified)
        supported = sum(1 for c in verified if c.supported)
        unsupported = total - supported

        if total > 0:
            overall_confidence = sum(c.confidence for c in verified) / total
            hallucination_rate = unsupported / total
        else:
            overall_confidence = 0.0
            hallucination_rate = 0.0

        # Step 4: Build grounded answer (only supported claims)
        grounded_parts = [c.text for c in verified if c.supported]
        grounded_answer = " ".join(grounded_parts) if grounded_parts else ""

        return VerificationResult(
            claims=verified,
            overall_confidence=overall_confidence,
            hallucination_rate=hallucination_rate,
            grounded_answer=grounded_answer,
            total_claims=total,
            supported_claims=supported,
            unsupported_claims=unsupported,
        )

    async def averify(self, answer: str, sources: list[str]) -> VerificationResult:
        """Async version of verify."""
        return await asyncio.to_thread(self.verify, answer, sources)
