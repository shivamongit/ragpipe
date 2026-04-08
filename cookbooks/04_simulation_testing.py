"""Cookbook 04: Retrieval Simulation — the 'pytest for RAG'.

Shows how to test your RAG pipeline against adversarial queries,
edge cases, and failure scenarios before deploying to production.
"""

from ragpipe.simulation import SimulationRunner, FailureScenario

# Simulate a simple retriever for demo purposes
class MockResult:
    def __init__(self, score):
        self.score = score
        class chunk:
            text = "Sample retrieved text"
            id = "mock:0"
        self.chunk = chunk()

def my_retrieve(query):
    """Your actual retriever goes here."""
    if not query or not query.strip():
        return []
    if len(query) > 1000:
        return [MockResult(0.2)]  # Low quality for very long queries
    return [MockResult(0.85), MockResult(0.72), MockResult(0.61)]

# 1. Create a simulation runner
sim = SimulationRunner(
    retrieve_fn=my_retrieve,
    min_retrieval_score=0.3,
    max_latency_ms=5000.0,
)

# 2. Add failure scenarios to test
sim.add_scenario(FailureScenario.ADVERSARIAL_QUERIES)
sim.add_scenario(FailureScenario.MISSING_CONTEXT)
sim.add_scenario(FailureScenario.EMPTY_RETRIEVAL)
sim.add_scenario(FailureScenario.AMBIGUOUS_QUERIES)
sim.add_scenario(FailureScenario.LONG_QUERIES)

# 3. Add custom test queries
sim.add_queries([
    "What is the revenue for Q4 2024?",
    "Compare product A vs product B",
    "Summarize the latest research paper",
], scenario="business_queries")

# 4. Add custom assertions
sim.add_assertion(lambda r: r.latency_ms < 2000)  # Must be under 2s

# 5. Run the simulation
result = sim.run(seed=42)

# 6. Review results
print(result.summary())
print(f"\nDetailed results:")
print(f"  Total: {result.total_queries}")
print(f"  Pass rate: {result.pass_rate:.0%}")
print(f"  Avg latency: {result.avg_latency_ms:.1f}ms")

# 7. Inspect failures
failures = [r for r in result.query_results if not r.passed]
if failures:
    print(f"\nFailures ({len(failures)}):")
    for f in failures:
        print(f"  [{f.scenario}] {f.query[:50]}... — {f.failure_reason}")
