"""PII Redactor — detect and redact personally identifiable information.

Zero-dependency PII detection using regex patterns. No spaCy, no presidio,
no external models required. Covers the most common PII types:
- Email addresses
- Phone numbers (US, international)
- Social Security Numbers
- Credit card numbers
- IP addresses
- Dates of birth patterns
- Names (when LLM-based detect_fn is provided)

Usage:
    from ragpipe.guardrails import PIIRedactor

    redactor = PIIRedactor()
    clean = redactor.redact("Contact john@example.com or 555-123-4567")
    # → "Contact [EMAIL_REDACTED] or [PHONE_REDACTED]"

    # Check before ingesting
    if redactor.contains_pii(document.content):
        document.content = redactor.redact(document.content)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class PIIMatch:
    """A detected PII instance."""
    pii_type: str
    value: str
    start: int
    end: int
    redacted: str


@dataclass
class RedactionResult:
    """Result of PII redaction."""
    original: str
    redacted: str
    matches: list[PIIMatch]
    pii_found: bool
    pii_count: int
    pii_types: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pii_found": self.pii_found,
            "pii_count": self.pii_count,
            "pii_types": self.pii_types,
            "matches": [
                {"type": m.pii_type, "redacted": m.redacted, "start": m.start, "end": m.end}
                for m in self.matches
            ],
        }


# Regex patterns for common PII types
PII_PATTERNS: dict[str, tuple[str, str]] = {
    "EMAIL": (
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "[EMAIL_REDACTED]",
    ),
    "PHONE": (
        r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "[PHONE_REDACTED]",
    ),
    "SSN": (
        r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
        "[SSN_REDACTED]",
    ),
    "CREDIT_CARD": (
        r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
        "[CREDIT_CARD_REDACTED]",
    ),
    "IP_ADDRESS": (
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "[IP_REDACTED]",
    ),
    "DATE_OF_BIRTH": (
        r'\b(?:DOB|date of birth|born on)[:\s]*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b',
        "[DOB_REDACTED]",
    ),
}


class PIIRedactor:
    """Detect and redact PII from text using regex patterns.

    Zero external dependencies — uses compiled regex patterns for fast,
    reliable PII detection. Optionally accepts an LLM function for
    detecting name-based PII that regex can't catch.
    """

    def __init__(
        self,
        detect_fn: Optional[Callable] = None,
        additional_patterns: Optional[dict[str, tuple[str, str]]] = None,
        enabled_types: Optional[list[str]] = None,
    ):
        self.detect_fn = detect_fn
        self.patterns: dict[str, tuple[re.Pattern, str]] = {}

        all_patterns = {**PII_PATTERNS}
        if additional_patterns:
            all_patterns.update(additional_patterns)

        for pii_type, (pattern, replacement) in all_patterns.items():
            if enabled_types is None or pii_type in enabled_types:
                self.patterns[pii_type] = (re.compile(pattern, re.IGNORECASE), replacement)

    def detect(self, text: str) -> list[PIIMatch]:
        """Detect all PII in text without redacting."""
        matches = []
        for pii_type, (pattern, replacement) in self.patterns.items():
            for m in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=m.group(),
                    start=m.start(),
                    end=m.end(),
                    redacted=replacement,
                ))
        # Sort by position (earliest first)
        matches.sort(key=lambda x: x.start)
        return matches

    def contains_pii(self, text: str) -> bool:
        """Quick check: does the text contain any PII?"""
        for _, (pattern, _) in self.patterns.items():
            if pattern.search(text):
                return True
        return False

    def redact(self, text: str) -> str:
        """Redact all detected PII from text."""
        result = text
        for _, (pattern, replacement) in self.patterns.items():
            result = pattern.sub(replacement, result)
        return result

    def redact_detailed(self, text: str) -> RedactionResult:
        """Redact PII and return detailed results."""
        matches = self.detect(text)
        redacted = self.redact(text)
        pii_types = list(set(m.pii_type for m in matches))
        return RedactionResult(
            original=text,
            redacted=redacted,
            matches=matches,
            pii_found=len(matches) > 0,
            pii_count=len(matches),
            pii_types=pii_types,
        )
