"""Cookbook 05: Agentic Retrieval — multi-step query decomposition + critique.

Shows how to use the AgenticPipeline for complex queries that require
multi-step retrieval, comparison, aggregation, and answer critique.
"""

from ragpipe.agents.planner import AgenticPipeline, RetrievalPlanner

# Simulate retrieval and generation for demo
class MockResult:
    def __init__(self, text, score):
        self.score = score
        class chunk:
            pass
        self.chunk = chunk()
        self.chunk.text = text
        self.chunk.id = f"chunk:{hash(text) % 1000}"
        self.chunk.metadata = {}

def my_retrieve(query):
    """Replace with your actual retriever."""
    return [
        MockResult(f"Result for: {query[:50]}", 0.85),
        MockResult(f"Additional context about: {query[:30]}", 0.72),
    ]

def my_generate(query, results):
    """Replace with your actual generator."""
    context = "\n".join(r.chunk.text for r in results)
    return f"Based on {len(results)} sources: Answer to '{query[:40]}...'"

def my_critique(query, answer, results):
    """Optional: critique the generated answer."""
    if len(results) < 3:
        return "Consider retrieving more sources for better coverage."
    return "Answer is well-supported by the retrieved sources."

# 1. Create the agentic pipeline
agent = AgenticPipeline(
    retrieve_fn=my_retrieve,
    generate_fn=my_generate,
    critique_fn=my_critique,
    max_rounds=3,
)

# 2. Simple query — single retrieval step
print("--- Simple Query ---")
result = agent.run("What is retrieval-augmented generation?")
print(result.summary())

# 3. Comparison query — auto-decomposes into parallel searches
print("\n--- Comparison Query ---")
result = agent.run("Compare FAISS and ChromaDB for vector search")
print(result.summary())

# 4. Aggregation query — search + aggregate
print("\n--- Aggregation Query ---")
result = agent.run("List all machine learning frameworks mentioned")
print(result.summary())

# 5. Inspect the plan
print("\n--- Query Plan Details ---")
planner = RetrievalPlanner()
plan = planner.plan("Compare Python and Rust for systems programming then summarize")
for step in plan.steps:
    print(f"  Step {step.step_id}: [{step.step_type.value}] {step.query}")
    if step.depends_on:
        print(f"    depends_on: {step.depends_on}")
print(f"  Reasoning: {plan.reasoning}")
