"""Tests for ragpipe.context.window — Context Engineering Primitives."""

from ragpipe.context.window import ContextWindow, ContextItem, ContextStrategy


# ── ContextItem ───────────────────────────────────────────────────────────────

def test_context_item_token_estimate():
    item = ContextItem(text="hello world this is a test")
    assert item.token_count > 0


def test_context_item_priority_defaults_to_score():
    item = ContextItem(text="test", score=0.85)
    assert item.priority == 0.85


# ── ContextWindow basics ─────────────────────────────────────────────────────

def test_context_window_empty():
    ctx = ContextWindow()
    assert ctx.item_count == 0
    assert ctx.total_tokens == 0
    assert ctx.utilization == 0.0


def test_context_window_add():
    ctx = ContextWindow()
    ctx.add("Hello world", score=0.9)
    ctx.add("Second item", score=0.7)
    assert ctx.item_count == 2


def test_context_window_add_items():
    items = [ContextItem(text="A", score=0.9), ContextItem(text="B", score=0.5)]
    ctx = ContextWindow()
    ctx.add_items(items)
    assert ctx.item_count == 2


def test_context_window_chaining():
    ctx = ContextWindow()
    result = ctx.add("test", score=0.9)
    assert result is ctx  # Returns self for chaining


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_deduplicate_exact():
    ctx = ContextWindow()
    ctx.add("Same text", score=0.9)
    ctx.add("Same text", score=0.8)
    ctx.add("Different text", score=0.7)
    ctx.deduplicate(method="exact")
    assert ctx.item_count == 2


def test_deduplicate_hash():
    ctx = ContextWindow()
    ctx.add("same content", score=0.9)
    ctx.add("same content", score=0.8)
    ctx.deduplicate(method="hash")
    assert ctx.item_count == 1


def test_deduplicate_jaccard():
    ctx = ContextWindow()
    ctx.add("the quick brown fox jumps over the lazy dog", score=0.9)
    ctx.add("the quick brown fox leaps over the lazy dog", score=0.8)  # very similar
    ctx.add("completely different unrelated content here", score=0.7)
    ctx.deduplicate(similarity_threshold=0.7, method="jaccard")
    assert ctx.item_count == 2  # Fox sentences merged


def test_deduplicate_no_duplicates():
    ctx = ContextWindow()
    ctx.add("Alpha topic content", score=0.9)
    ctx.add("Beta different subject", score=0.8)
    ctx.deduplicate(method="exact")
    assert ctx.item_count == 2


# ── Prioritization ────────────────────────────────────────────────────────────

def test_prioritize_relevance():
    ctx = ContextWindow()
    ctx.add("Low", score=0.3)
    ctx.add("High", score=0.9)
    ctx.add("Mid", score=0.6)
    ctx.prioritize("relevance")
    items = ctx.items
    assert items[0].score == 0.9
    assert items[2].score == 0.3


def test_prioritize_density():
    ctx = ContextWindow()
    ctx.add("the the the the the the the the the the", score=0.5)  # low density: 1 unique / 10 tokens
    ctx.add("alpha beta gamma delta epsilon zeta eta theta", score=0.5)  # high density: 8 unique / 8 tokens
    ctx.prioritize("density")
    # Higher density (more unique words per token) should be first
    assert "alpha" in ctx.items[0].text


def test_prioritize_diversity():
    ctx = ContextWindow()
    ctx.add("cats dogs pets animals", score=0.9)
    ctx.add("cats dogs pets mammals", score=0.85)  # Similar to first
    ctx.add("programming code software engineering", score=0.7)  # Different
    ctx.prioritize("diversity")
    # Should interleave diverse topics
    assert ctx.item_count == 3


def test_prioritize_position():
    ctx = ContextWindow()
    ctx.add("B", score=0.5, chunk_id="doc:2")
    ctx.add("A", score=0.9, chunk_id="doc:1")
    ctx.prioritize("position")
    assert ctx.items[0].chunk_id == "doc:1"


# ── Compression ───────────────────────────────────────────────────────────────

def test_compress_truncation():
    ctx = ContextWindow()
    ctx.add("A" * 10000, score=0.9)
    ctx.compress(max_item_tokens=50)
    assert ctx.items[0].token_count <= 50


def test_compress_with_fn():
    ctx = ContextWindow()
    ctx.add("This is a very long text that should be summarized", score=0.9)
    ctx.compress(compress_fn=lambda t: "summary", max_item_tokens=1)
    assert ctx.items[0].text == "summary"


# ── Budgeting ─────────────────────────────────────────────────────────────────

def test_budget_fits():
    ctx = ContextWindow(max_tokens=10000)
    ctx.add("Short text", score=0.9)
    ctx.budget()
    assert ctx.item_count == 1


def test_budget_removes_excess():
    ctx = ContextWindow(max_tokens=20)
    ctx.add("A" * 100, score=0.9)
    ctx.add("B" * 100, score=0.8)
    ctx.add("C" * 100, score=0.7)
    ctx.budget(max_tokens=30)
    assert ctx.item_count < 3


# ── Filtering ─────────────────────────────────────────────────────────────────

def test_filter_by_score():
    ctx = ContextWindow()
    ctx.add("A", score=0.9)
    ctx.add("B", score=0.3)
    ctx.add("C", score=0.1)
    ctx.filter_by_score(min_score=0.5)
    assert ctx.item_count == 1
    assert ctx.items[0].score == 0.9


def test_filter_by_source():
    ctx = ContextWindow()
    ctx.add("A", score=0.9, source="wiki")
    ctx.add("B", score=0.8, source="blog")
    ctx.filter_by_source(["wiki"])
    assert ctx.item_count == 1


def test_filter_predicate():
    ctx = ContextWindow()
    ctx.add("short", score=0.9)
    ctx.add("this is a much longer piece of text content", score=0.8)
    ctx.filter(lambda item: len(item.text) > 10)
    assert ctx.item_count == 1


# ── Rendering ─────────────────────────────────────────────────────────────────

def test_render_default():
    ctx = ContextWindow()
    ctx.add("First chunk", score=0.9)
    ctx.add("Second chunk", score=0.7)
    rendered = ctx.render()
    assert "Source 1" in rendered
    assert "First chunk" in rendered


def test_render_custom_format():
    ctx = ContextWindow()
    ctx.add("Content", score=0.95)
    rendered = ctx.render(format="[{i}] {text}")
    assert rendered == "[1] Content"


def test_render_citations():
    ctx = ContextWindow()
    ctx.add("Text A", score=0.9, source="doc1")
    citations = ctx.render_citations()
    assert len(citations) == 1
    assert citations[0]["source"] == "doc1"


# ── Chaining / pipe ──────────────────────────────────────────────────────────

def test_pipe_operations():
    ctx = ContextWindow(max_tokens=5000)
    ctx.add("A", score=0.9)
    ctx.add("A", score=0.8)
    ctx.add("B", score=0.7)
    ctx.pipe(
        lambda c: c.deduplicate(method="exact"),
        lambda c: c.prioritize("relevance"),
    )
    assert ctx.item_count == 2


# ── Serialization ─────────────────────────────────────────────────────────────

def test_to_dict():
    ctx = ContextWindow()
    ctx.add("Test", score=0.9)
    d = ctx.to_dict()
    assert d["item_count"] == 1
    assert "utilization" in d


def test_summary():
    ctx = ContextWindow(max_tokens=1000)
    ctx.add("Test content here", score=0.85)
    s = ctx.summary()
    assert "ContextWindow" in s
    assert "1 items" in s


def test_clear():
    ctx = ContextWindow()
    ctx.add("A", score=0.9)
    ctx.clear()
    assert ctx.item_count == 0
    assert len(ctx.operations) == 0


# ── Properties ────────────────────────────────────────────────────────────────

def test_utilization():
    ctx = ContextWindow(max_tokens=100)
    ctx.add("A" * 200, score=0.9)  # ~50 tokens
    assert 0.0 < ctx.utilization <= 1.0


def test_operations_log():
    ctx = ContextWindow()
    ctx.add("A", score=0.9)
    ctx.add("A", score=0.8)
    ctx.deduplicate(method="exact")
    ctx.prioritize("relevance")
    assert len(ctx.operations) == 2
