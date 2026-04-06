"""Cookbook 06: Pipeline DAG — branching, conditional, and parallel RAG workflows.

Shows how to build complex RAG pipelines using a directed acyclic graph
instead of a linear chain. Supports fan-out, fan-in, conditional routing,
and error handling per node.
"""

from ragpipe.pipeline import PipelineDAG, Node, NodeType

# 1. Define node functions
def embed(inputs):
    query = inputs.get("query", "")
    return {"query": query, "embedding": [0.1, 0.2, 0.3]}  # placeholder

def retrieve_dense(inputs):
    return {"results": [f"dense_result_for_{inputs['query'][:20]}"], "source": "dense"}

def retrieve_sparse(inputs):
    return {"results": [f"sparse_result_for_{inputs['query'][:20]}"], "source": "sparse"}

def merge_results(inputs):
    """RRF-style merge of parallel retrieval branches."""
    all_results = []
    for branch_name, branch_output in inputs.items():
        if isinstance(branch_output, dict) and "results" in branch_output:
            all_results.extend(branch_output["results"])
    return {"merged_results": all_results, "count": len(all_results)}

def generate(inputs):
    results = inputs.get("merged_results", [])
    return {"answer": f"Generated from {len(results)} sources", "sources": results}

# 2. Build the DAG
dag = PipelineDAG()

# Add nodes
dag.add_node(Node("embed", NodeType.TRANSFORM, fn=embed))
dag.add_node(Node("dense", NodeType.RETRIEVE, fn=retrieve_dense))
dag.add_node(Node("sparse", NodeType.RETRIEVE, fn=retrieve_sparse))
dag.add_node(Node("merge", NodeType.MERGE, fn=merge_results))
dag.add_node(Node("generate", NodeType.GENERATE, fn=generate))

# Wire the graph: embed → [dense, sparse] → merge → generate
dag.add_edge("embed", "dense")
dag.add_edge("embed", "sparse")
dag.add_edge("dense", "merge")
dag.add_edge("sparse", "merge")
dag.add_edge("merge", "generate")

# 3. Visualize the DAG
print(dag.summary())

# 4. Execute
result = dag.execute({"query": "What is knowledge graph RAG?"})
print(f"\nOutput: {result.output}")
print(f"\n{result.summary()}")

# 5. Conditional execution example
print("\n--- Conditional DAG ---")
cdag = PipelineDAG()
cdag.add_node(Node("classify", fn=lambda x: {**x, "is_complex": len(x.get("query", "")) > 30}))
cdag.add_node(Node("simple_search", fn=lambda x: {"answer": "simple result"},
                    condition_fn=lambda x: not x.get("is_complex", False)))
cdag.add_node(Node("deep_search", fn=lambda x: {"answer": "deep multi-hop result"},
                    condition_fn=lambda x: x.get("is_complex", False)))

cdag.add_edge("classify", "simple_search")
cdag.add_edge("classify", "deep_search")

# Short query → simple path
r1 = cdag.execute({"query": "What is RAG?"})
print(f"Short query result: {r1.output}")

# Long query → deep path
r2 = cdag.execute({"query": "Compare and contrast different retrieval augmentation strategies for multi-hop reasoning"})
print(f"Long query result: {r2.output}")
