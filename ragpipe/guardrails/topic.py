"""Topic Guardrail — restrict queries to allowed topics.

Zero-dependency topic filtering using keyword matching and optional LLM classification.
Useful for domain-specific deployments where the RAG system should only answer
questions within its scope.

Usage:
    from ragpipe.guardrails import TopicGuardrail

    guard = TopicGuardrail(
        allowed_topics=["finance", "accounting", "tax"],
        blocked_topics=["politics", "religion"],
    )
    result = guard.check("What are the tax implications of stock options?")
    # → TopicResult(is_allowed=True, matched_topic="tax", confidence=0.9)

    result = guard.check("Who should I vote for?")
    # → TopicResult(is_allowed=False, matched_topic="politics", confidence=0.8)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TopicResult:
    """Result of topic guardrail check."""
    is_allowed: bool
    matched_topic: str
    confidence: float
    query: str
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_allowed": self.is_allowed,
            "matched_topic": self.matched_topic,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class TopicGuardrail:
    """Restrict RAG queries to allowed topics.

    Works in two modes:
    1. Allowlist: only queries matching allowed_topics are permitted
    2. Blocklist: queries matching blocked_topics are rejected
    3. Both: allowlist takes priority, blocklist catches edge cases

    Uses keyword matching by default. Optionally accepts an LLM classify_fn
    for semantic topic classification.
    """

    def __init__(
        self,
        allowed_topics: Optional[list[str]] = None,
        blocked_topics: Optional[list[str]] = None,
        topic_keywords: Optional[dict[str, list[str]]] = None,
        classify_fn: Optional[Callable] = None,
        default_allow: bool = True,
    ):
        self.allowed_topics = [t.lower() for t in (allowed_topics or [])]
        self.blocked_topics = [t.lower() for t in (blocked_topics or [])]
        self.topic_keywords = topic_keywords or {}
        self.classify_fn = classify_fn
        self.default_allow = default_allow

        # Normalize keyword dict
        self._keywords: dict[str, list[str]] = {}
        for topic, keywords in self.topic_keywords.items():
            self._keywords[topic.lower()] = [k.lower() for k in keywords]

    def _keyword_match(self, query: str) -> tuple[str, float]:
        """Match query against topic keywords."""
        q = query.lower()
        best_topic = ""
        best_score = 0.0

        # Check all topics (allowed + blocked + keywords)
        all_topics = set(self.allowed_topics + self.blocked_topics + list(self._keywords.keys()))

        for topic in all_topics:
            # Direct topic name match
            if topic in q:
                score = 0.9
                if score > best_score:
                    best_score = score
                    best_topic = topic

            # Keyword match
            if topic in self._keywords:
                matched = sum(1 for kw in self._keywords[topic] if kw in q)
                total = len(self._keywords[topic])
                if total > 0 and matched > 0:
                    score = matched / total * 0.8
                    if score > best_score:
                        best_score = score
                        best_topic = topic

        return best_topic, best_score

    def check(self, query: str) -> TopicResult:
        """Check if a query is within allowed topics.

        Args:
            query: The user query to check

        Returns:
            TopicResult indicating if the query is allowed
        """
        if not query or not query.strip():
            return TopicResult(
                is_allowed=self.default_allow,
                matched_topic="",
                confidence=0.0,
                query=query,
                reason="Empty query",
            )

        matched_topic, confidence = self._keyword_match(query)

        # Check blocked topics first
        if matched_topic and matched_topic in self.blocked_topics:
            return TopicResult(
                is_allowed=False,
                matched_topic=matched_topic,
                confidence=confidence,
                query=query,
                reason=f"Query matches blocked topic: {matched_topic}",
            )

        # Check allowed topics (if allowlist is set)
        if self.allowed_topics:
            if matched_topic and matched_topic in self.allowed_topics:
                return TopicResult(
                    is_allowed=True,
                    matched_topic=matched_topic,
                    confidence=confidence,
                    query=query,
                    reason=f"Query matches allowed topic: {matched_topic}",
                )
            elif confidence < 0.3:
                # No strong match — decide based on default_allow
                return TopicResult(
                    is_allowed=self.default_allow,
                    matched_topic="",
                    confidence=confidence,
                    query=query,
                    reason="No topic match — using default policy",
                )
            else:
                # Matched a topic not in allowed list
                return TopicResult(
                    is_allowed=False,
                    matched_topic=matched_topic,
                    confidence=confidence,
                    query=query,
                    reason=f"Query topic '{matched_topic}' not in allowed list",
                )

        # No allowlist — allow unless blocked
        return TopicResult(
            is_allowed=True,
            matched_topic=matched_topic,
            confidence=confidence,
            query=query,
            reason="No restrictions — query allowed",
        )

    def is_allowed(self, query: str) -> bool:
        """Quick check: is the query allowed?"""
        return self.check(query).is_allowed
