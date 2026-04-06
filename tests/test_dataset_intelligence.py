"""Tests for ragpipe.intelligence — Dataset Intelligence Layer."""

import time
from ragpipe.intelligence.analyzer import (
    DatasetAnalyzer, DatasetReport, DocumentIssue,
    IssueType, IssueSeverity, DatasetStats,
)


# ── DocumentIssue ─────────────────────────────────────────────────────────────

def test_document_issue_to_dict():
    issue = DocumentIssue(
        issue_type=IssueType.DUPLICATE,
        severity=IssueSeverity.HIGH,
        description="Duplicate found",
        doc_ids=["a", "b"],
    )
    d = issue.to_dict()
    assert d["type"] == "duplicate"
    assert d["severity"] == "high"


# ── DatasetStats ──────────────────────────────────────────────────────────────

def test_dataset_stats_to_dict():
    stats = DatasetStats(document_count=10, total_chars=5000, unique_words=200)
    d = stats.to_dict()
    assert d["document_count"] == 10


# ── DatasetReport ─────────────────────────────────────────────────────────────

def test_report_empty():
    report = DatasetReport()
    assert report.issue_count == 0
    assert report.health_score == 1.0


def test_report_issues_by_type():
    report = DatasetReport(issues=[
        DocumentIssue(IssueType.DUPLICATE, IssueSeverity.HIGH, "dup"),
        DocumentIssue(IssueType.DUPLICATE, IssueSeverity.HIGH, "dup2"),
        DocumentIssue(IssueType.STALE, IssueSeverity.MEDIUM, "old"),
    ])
    by_type = report.issues_by_type
    assert by_type["duplicate"] == 2
    assert by_type["stale"] == 1


def test_report_critical_issues():
    report = DatasetReport(issues=[
        DocumentIssue(IssueType.EMPTY_DOCUMENT, IssueSeverity.CRITICAL, "empty"),
        DocumentIssue(IssueType.STALE, IssueSeverity.MEDIUM, "old"),
    ])
    assert len(report.critical_issues) == 1


def test_report_summary():
    report = DatasetReport(
        stats=DatasetStats(document_count=5, total_chars=1000, unique_words=50),
        health_score=0.85,
    )
    s = report.summary()
    assert "85%" in s
    assert "5 docs" in s


def test_report_to_dict():
    report = DatasetReport(health_score=0.9)
    d = report.to_dict()
    assert d["health_score"] == 0.9
    assert "stats" in d


# ── DatasetAnalyzer basics ────────────────────────────────────────────────────

def test_analyzer_empty():
    analyzer = DatasetAnalyzer()
    report = analyzer.analyze()
    assert report.stats.document_count == 0
    assert report.health_score == 0.0


def test_analyzer_add_strings():
    analyzer = DatasetAnalyzer()
    analyzer.add_documents(["Hello world, this is a test document with enough content to pass."])
    report = analyzer.analyze()
    assert report.stats.document_count == 1


def test_analyzer_add_dicts():
    analyzer = DatasetAnalyzer()
    analyzer.add_documents([
        {"content": "First document with enough content to be valid here.", "doc_id": "d1", "metadata": {}},
        {"content": "Second document also has sufficient content here.", "doc_id": "d2", "metadata": {}},
    ])
    report = analyzer.analyze()
    assert report.stats.document_count == 2


# ── Empty / short detection ───────────────────────────────────────────────────

def test_detect_empty_document():
    analyzer = DatasetAnalyzer()
    analyzer.add_documents([{"content": "", "doc_id": "empty", "metadata": {}}])
    report = analyzer.analyze()
    types = [i.issue_type for i in report.issues]
    assert IssueType.EMPTY_DOCUMENT in types


def test_detect_short_document():
    analyzer = DatasetAnalyzer(min_doc_length=100)
    analyzer.add_documents([{"content": "Too short.", "doc_id": "short", "metadata": {}}])
    report = analyzer.analyze()
    types = [i.issue_type for i in report.issues]
    assert IssueType.SHORT_DOCUMENT in types


def test_no_issue_for_normal_length():
    analyzer = DatasetAnalyzer(min_doc_length=10)
    analyzer.add_documents([{
        "content": "This is a perfectly normal document with enough content.",
        "doc_id": "ok",
        "metadata": {},
    }])
    report = analyzer.analyze()
    empty_short = [i for i in report.issues if i.issue_type in (IssueType.EMPTY_DOCUMENT, IssueType.SHORT_DOCUMENT)]
    assert len(empty_short) == 0


# ── Duplicate detection ───────────────────────────────────────────────────────

def test_detect_exact_duplicate():
    content = "This is an exact duplicate document with sufficient content."
    analyzer = DatasetAnalyzer()
    analyzer.add_documents([
        {"content": content, "doc_id": "a", "metadata": {}},
        {"content": content, "doc_id": "b", "metadata": {}},
    ])
    report = analyzer.analyze()
    dup_issues = [i for i in report.issues if i.issue_type == IssueType.DUPLICATE]
    assert len(dup_issues) == 1
    assert set(dup_issues[0].doc_ids) == {"a", "b"}


def test_detect_near_duplicate():
    analyzer = DatasetAnalyzer(duplicate_threshold=0.8)
    analyzer.add_documents([
        {"content": "the quick brown fox jumps over the lazy dog in the park", "doc_id": "a", "metadata": {}},
        {"content": "the quick brown fox leaps over the lazy dog in the park", "doc_id": "b", "metadata": {}},
    ])
    report = analyzer.analyze()
    near_dup = [i for i in report.issues if i.issue_type == IssueType.NEAR_DUPLICATE]
    assert len(near_dup) >= 1


def test_no_false_duplicate():
    analyzer = DatasetAnalyzer()
    analyzer.add_documents([
        {"content": "Document about machine learning algorithms and neural networks.", "doc_id": "a", "metadata": {}},
        {"content": "Recipe for chocolate cake with butter and sugar ingredients.", "doc_id": "b", "metadata": {}},
    ])
    report = analyzer.analyze()
    dup_issues = [i for i in report.issues if i.issue_type in (IssueType.DUPLICATE, IssueType.NEAR_DUPLICATE)]
    assert len(dup_issues) == 0


# ── Stale detection ───────────────────────────────────────────────────────────

def test_detect_stale_document():
    old_time = time.time() - (400 * 86400)  # 400 days ago
    analyzer = DatasetAnalyzer(stale_days=365)
    analyzer.add_documents([{
        "content": "This is an old document that should be flagged as stale.",
        "doc_id": "old",
        "metadata": {"timestamp": old_time},
    }])
    report = analyzer.analyze()
    stale = [i for i in report.issues if i.issue_type == IssueType.STALE]
    assert len(stale) == 1


def test_no_stale_for_recent():
    analyzer = DatasetAnalyzer(stale_days=365)
    analyzer.add_documents([{
        "content": "This is a recent document that should not be flagged.",
        "doc_id": "recent",
        "metadata": {"timestamp": time.time()},
    }])
    report = analyzer.analyze()
    stale = [i for i in report.issues if i.issue_type == IssueType.STALE]
    assert len(stale) == 0


# ── Low quality detection ─────────────────────────────────────────────────────

def test_detect_repetitive_content():
    analyzer = DatasetAnalyzer()
    # Very repetitive: same word repeated many times
    content = " ".join(["spam"] * 100)
    analyzer.add_documents([{"content": content, "doc_id": "rep", "metadata": {}}])
    report = analyzer.analyze()
    low_q = [i for i in report.issues if i.issue_type == IssueType.LOW_QUALITY]
    assert len(low_q) == 1


def test_no_low_quality_for_diverse():
    analyzer = DatasetAnalyzer()
    content = "Machine learning algorithms include decision trees random forests neural networks support vector machines gradient boosting and many other sophisticated approaches"
    analyzer.add_documents([{"content": content, "doc_id": "diverse", "metadata": {}}])
    report = analyzer.analyze()
    low_q = [i for i in report.issues if i.issue_type == IssueType.LOW_QUALITY]
    assert len(low_q) == 0


# ── Health score ──────────────────────────────────────────────────────────────

def test_health_score_perfect():
    analyzer = DatasetAnalyzer(min_doc_length=10)
    analyzer.add_documents([
        {"content": "A wonderful varied document about technology and science.", "doc_id": "a", "metadata": {}},
        {"content": "Another great unique piece about history and culture.", "doc_id": "b", "metadata": {}},
    ])
    report = analyzer.analyze()
    assert report.health_score >= 0.9


def test_health_score_degraded():
    analyzer = DatasetAnalyzer()
    # Add some bad documents
    analyzer.add_documents([
        {"content": "", "doc_id": "empty1", "metadata": {}},
        {"content": "", "doc_id": "empty2", "metadata": {}},
        {"content": "ok document with enough content to be valid here.", "doc_id": "ok", "metadata": {}},
    ])
    report = analyzer.analyze()
    assert report.health_score < 1.0


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_stats_computed():
    analyzer = DatasetAnalyzer()
    analyzer.add_documents([
        {"content": "First document with some content here.", "doc_id": "a", "metadata": {}},
        {"content": "Second document with different words.", "doc_id": "b", "metadata": {}},
    ])
    report = analyzer.analyze()
    assert report.stats.document_count == 2
    assert report.stats.total_chars > 0
    assert report.stats.avg_doc_length > 0
    assert report.stats.unique_words > 0
    assert report.stats.vocabulary_richness > 0


def test_stats_top_terms():
    analyzer = DatasetAnalyzer()
    analyzer.add_documents([
        {"content": "Python programming Python code Python language", "doc_id": "a", "metadata": {}},
    ])
    report = analyzer.analyze()
    top_terms = dict(report.stats.top_terms)
    assert "python" in top_terms
    assert top_terms["python"] >= 3


# ── Clear ─────────────────────────────────────────────────────────────────────

def test_analyzer_clear():
    analyzer = DatasetAnalyzer()
    analyzer.add_documents(["Some content"])
    analyzer.clear()
    report = analyzer.analyze()
    assert report.stats.document_count == 0
