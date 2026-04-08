"""Cookbook 02: Context Engineering — programmable context composition.

Shows how to use ContextWindow to build optimal LLM context beyond
simple "top-K stuffing". Demonstrates deduplication, prioritization,
compression, filtering, and token budgeting.
"""

from ragpipe.context import ContextWindow, ContextItem

# Simulate retrieval results as ContextItems
items = [
    ContextItem(text="RAG combines retrieval with generation for accurate answers.", score=0.95, source="intro.md"),
    ContextItem(text="RAG combines retrieval with generation for precise answers.", score=0.90, source="intro_v2.md"),  # near-dup
    ContextItem(text="Vector databases store embeddings as numerical arrays.", score=0.85, source="vectors.md"),
    ContextItem(text="LLMs can hallucinate when lacking factual context.", score=0.80, source="hallucination.md"),
    ContextItem(text="Chunking splits documents into manageable pieces.", score=0.70, source="chunking.md"),
    ContextItem(text="Fine-tuning adapts models to specific domains.", score=0.40, source="fine_tuning.md"),
    ContextItem(text="Python is a popular programming language.", score=0.15, source="python.md"),  # irrelevant
]

# Build an optimized context window
ctx = ContextWindow(max_tokens=500)
ctx.add_items(items)

print(f"Before: {ctx.item_count} items, {ctx.total_tokens} tokens")

# Chain operations for optimal context
ctx.deduplicate(similarity_threshold=0.8)      # Remove near-duplicates
ctx.filter_by_score(min_score=0.3)             # Drop low-relevance items
ctx.prioritize("relevance")                     # Sort by score
ctx.budget(max_tokens=400)                      # Enforce token limit

print(f"After:  {ctx.item_count} items, {ctx.total_tokens} tokens")
print(f"Utilization: {ctx.utilization:.0%}")
print(f"Operations: {' → '.join(ctx.operations)}")

# Render for LLM prompt
prompt_context = ctx.render()
print(f"\n--- Rendered Context ---\n{prompt_context}")

# Get structured citations
citations = ctx.render_citations()
print(f"\n--- Citations ---")
for c in citations:
    print(f"  [{c['index']}] {c['source']} (score={c['score']:.2f})")

# Full summary
print(f"\n{ctx.summary()}")
