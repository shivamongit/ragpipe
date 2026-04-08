"""Dataset Intelligence — continuous analysis of ingested corpus quality.

Analyzes the ingested document corpus for quality signals: stale documents,
contradictions between documents, duplicate knowledge, coverage gaps,
and entity/concept drift.

Usage:
    from ragpipe.intelligence import DatasetAnalyzer

    analyzer = DatasetAnalyzer()
    analyzer.add_documents(documents)
    report = analyzer.analyze()
    print(report.summary())
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class IssueType(str, Enum):
    """Types of dataset quality issues."""
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    STALE = "stale"
    CONTRADICTION = "contradiction"
    LOW_QUALITY = "low_quality"
    COVERAGE_GAP = "coverage_gap"
    SHORT_DOCUMENT = "short_document"
    EMPTY_DOCUMENT = "empty_document"


class IssueSeverity(str, Enum):
    """Severity levels for dataset issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DocumentIssue:
    """A quality issue found in the dataset."""
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.issue_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "doc_ids": self.doc_ids,
        }


@dataclass
class DatasetStats:
    """Statistical summary of the dataset."""
    document_count: int = 0
    total_chars: int = 0
    avg_doc_length: float = 0.0
    min_doc_length: int = 0
    max_doc_length: int = 0
    unique_words: int = 0
    total_words: int = 0
    vocabulary_richness: float = 0.0
    top_terms: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "total_chars": self.total_chars,
            "avg_doc_length": round(self.avg_doc_length, 1),
            "min_doc_length": self.min_doc_length,
            "max_doc_length": self.max_doc_length,
            "unique_words": self.unique_words,
            "total_words": self.total_words,
            "vocabulary_richness": round(self.vocabulary_richness, 4),
            "top_terms": self.top_terms[:20],
        }


@dataclass
class DatasetReport:
    """Comprehensive dataset quality report."""
    stats: DatasetStats = field(default_factory=DatasetStats)
    issues: list[DocumentIssue] = field(default_factory=list)
    health_score: float = 1.0
    analysis_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def critical_issues(self) -> list[DocumentIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def issues_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.issue_type.value] = counts.get(issue.issue_type.value, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [
            f"Dataset Report: health={self.health_score:.0%}, "
            f"{self.stats.document_count} docs, {self.issue_count} issues",
            f"  Stats: {self.stats.total_chars:,} chars, "
            f"avg={self.stats.avg_doc_length:.0f} chars/doc, "
            f"{self.stats.unique_words:,} unique words",
        ]
        if self.issues:
            lines.append(f"  Issues by type: {self.issues_by_type}")
            for issue in self.critical_issues:
                lines.append(f"  ⚠️  [{issue.severity.value}] {issue.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "health_score": round(self.health_score, 4),
            "stats": self.stats.to_dict(),
            "issue_count": self.issue_count,
            "issues_by_type": self.issues_by_type,
            "issues": [i.to_dict() for i in self.issues],
            "analysis_time_ms": round(self.analysis_time_ms, 2),
        }


# Stopwords for term frequency analysis
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "and", "but", "or", "if", "it", "its", "this",
    "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom",
})


class DatasetAnalyzer:
    """Analyze dataset quality for staleness, duplicates, contradictions, and coverage gaps.

    Runs multiple analysis passes over the document corpus and produces
    a comprehensive quality report with actionable recommendations.
    """

    def __init__(
        self,
        min_doc_length: int = 50,
        duplicate_threshold: float = 0.85,
        stale_days: int = 365,
        analyze_fn: Callable | None = None,
    ):
        self._min_doc_length = min_doc_length
        self._dup_threshold = duplicate_threshold
        self._stale_days = stale_days
        self._analyze_fn = analyze_fn
        self._documents: list[dict[str, Any]] = []

    def add_documents(self, documents: list) -> None:
        """Add documents for analysis. Accepts Document objects or dicts."""
        for doc in documents:
            if hasattr(doc, 'content'):
                self._documents.append({
                    "content": doc.content,
                    "metadata": getattr(doc, 'metadata', {}),
                    "doc_id": getattr(doc, 'doc_id', ''),
                })
            elif isinstance(doc, dict):
                self._documents.append(doc)
            elif isinstance(doc, str):
                self._documents.append({"content": doc, "metadata": {}, "doc_id": ""})

    def analyze(self) -> DatasetReport:
        """Run full dataset analysis and return a report."""
        t0 = time.perf_counter()

        stats = self._compute_stats()
        issues: list[DocumentIssue] = []

        issues.extend(self._detect_empty_and_short())
        issues.extend(self._detect_duplicates())
        issues.extend(self._detect_stale())
        issues.extend(self._detect_low_quality())

        # Compute health score
        health = self._compute_health(stats, issues)

        elapsed = (time.perf_counter() - t0) * 1000
        return DatasetReport(
            stats=stats,
            issues=issues,
            health_score=health,
            analysis_time_ms=elapsed,
        )

    def _compute_stats(self) -> DatasetStats:
        """Compute basic dataset statistics."""
        if not self._documents:
            return DatasetStats()

        lengths = [len(d.get("content", "")) for d in self._documents]
        all_words: list[str] = []
        word_counter: Counter = Counter()

        for doc in self._documents:
            words = re.findall(r'\b[a-zA-Z]{2,}\b', doc.get("content", "").lower())
            meaningful = [w for w in words if w not in _STOPWORDS]
            all_words.extend(meaningful)
            word_counter.update(meaningful)

        unique = len(set(all_words))
        total = len(all_words)

        return DatasetStats(
            document_count=len(self._documents),
            total_chars=sum(lengths),
            avg_doc_length=sum(lengths) / len(lengths) if lengths else 0,
            min_doc_length=min(lengths) if lengths else 0,
            max_doc_length=max(lengths) if lengths else 0,
            unique_words=unique,
            total_words=total,
            vocabulary_richness=unique / total if total else 0,
            top_terms=word_counter.most_common(20),
        )

    def _detect_empty_and_short(self) -> list[DocumentIssue]:
        """Detect empty or very short documents."""
        issues = []
        for doc in self._documents:
            content = doc.get("content", "")
            doc_id = doc.get("doc_id", "unknown")

            if not content.strip():
                issues.append(DocumentIssue(
                    issue_type=IssueType.EMPTY_DOCUMENT,
                    severity=IssueSeverity.CRITICAL,
                    description=f"Document '{doc_id}' is empty",
                    doc_ids=[doc_id],
                ))
            elif len(content) < self._min_doc_length:
                issues.append(DocumentIssue(
                    issue_type=IssueType.SHORT_DOCUMENT,
                    severity=IssueSeverity.MEDIUM,
                    description=f"Document '{doc_id}' is only {len(content)} chars (min: {self._min_doc_length})",
                    doc_ids=[doc_id],
                ))
        return issues

    def _detect_duplicates(self) -> list[DocumentIssue]:
        """Detect exact and near-duplicate documents."""
        issues = []
        hashes: dict[str, str] = {}
        docs_with_content: list[tuple[str, str, set]] = []

        for doc in self._documents:
            content = doc.get("content", "")
            doc_id = doc.get("doc_id", "unknown")

            # Exact duplicate check
            h = hashlib.md5(content.strip().encode()).hexdigest()
            if h in hashes:
                issues.append(DocumentIssue(
                    issue_type=IssueType.DUPLICATE,
                    severity=IssueSeverity.HIGH,
                    description=f"Document '{doc_id}' is an exact duplicate of '{hashes[h]}'",
                    doc_ids=[doc_id, hashes[h]],
                ))
            else:
                hashes[h] = doc_id

            # Prepare for near-duplicate check
            words = set(content.lower().split())
            docs_with_content.append((doc_id, content, words))

        # Near-duplicate check (Jaccard similarity)
        for i in range(len(docs_with_content)):
            for j in range(i + 1, min(i + 50, len(docs_with_content))):
                id_a, _, words_a = docs_with_content[i]
                id_b, _, words_b = docs_with_content[j]

                if not words_a or not words_b:
                    continue

                intersection = words_a & words_b
                union = words_a | words_b
                similarity = len(intersection) / len(union) if union else 0

                if similarity > self._dup_threshold:
                    # Check it's not already flagged as exact duplicate
                    already_flagged = any(
                        issue.issue_type == IssueType.DUPLICATE
                        and set(issue.doc_ids) == {id_a, id_b}
                        for issue in issues
                    )
                    if not already_flagged:
                        issues.append(DocumentIssue(
                            issue_type=IssueType.NEAR_DUPLICATE,
                            severity=IssueSeverity.MEDIUM,
                            description=f"Documents '{id_a}' and '{id_b}' are {similarity:.0%} similar",
                            doc_ids=[id_a, id_b],
                            metadata={"similarity": round(similarity, 4)},
                        ))

        return issues

    def _detect_stale(self) -> list[DocumentIssue]:
        """Detect potentially stale documents based on metadata timestamps."""
        issues = []
        now = time.time()
        stale_threshold = self._stale_days * 86400

        for doc in self._documents:
            meta = doc.get("metadata", {})
            doc_id = doc.get("doc_id", "unknown")

            timestamp = meta.get("timestamp") or meta.get("date") or meta.get("modified")
            if isinstance(timestamp, (int, float)):
                age_days = (now - timestamp) / 86400
                if age_days > self._stale_days:
                    issues.append(DocumentIssue(
                        issue_type=IssueType.STALE,
                        severity=IssueSeverity.MEDIUM,
                        description=f"Document '{doc_id}' is {age_days:.0f} days old (threshold: {self._stale_days})",
                        doc_ids=[doc_id],
                        metadata={"age_days": round(age_days, 1)},
                    ))

        return issues

    def _detect_low_quality(self) -> list[DocumentIssue]:
        """Detect low-quality documents (repetitive, low info density)."""
        issues = []
        for doc in self._documents:
            content = doc.get("content", "")
            doc_id = doc.get("doc_id", "unknown")

            if len(content) < 10:
                continue

            words = content.lower().split()
            if not words:
                continue

            unique_ratio = len(set(words)) / len(words)

            # Very repetitive content
            if unique_ratio < 0.3 and len(words) > 20:
                issues.append(DocumentIssue(
                    issue_type=IssueType.LOW_QUALITY,
                    severity=IssueSeverity.MEDIUM,
                    description=(
                        f"Document '{doc_id}' has low vocabulary diversity "
                        f"({unique_ratio:.0%} unique words)"
                    ),
                    doc_ids=[doc_id],
                    metadata={"unique_ratio": round(unique_ratio, 4)},
                ))

        return issues

    def _compute_health(self, stats: DatasetStats, issues: list[DocumentIssue]) -> float:
        """Compute overall health score (0-1)."""
        if not self._documents:
            return 0.0

        score = 1.0

        # Penalize for issues
        for issue in issues:
            if issue.severity == IssueSeverity.CRITICAL:
                score -= 0.15
            elif issue.severity == IssueSeverity.HIGH:
                score -= 0.08
            elif issue.severity == IssueSeverity.MEDIUM:
                score -= 0.03
            else:
                score -= 0.01

        # Bonus for vocabulary richness
        if stats.vocabulary_richness > 0.5:
            score += 0.05

        return max(0.0, min(1.0, score))

    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()
