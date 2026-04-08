"""Pipeline DAG — directed acyclic graph for complex RAG workflows.

Replaces the linear pipeline with a graph that supports branching,
conditional steps, parallel execution, and merging.

Usage:
    from ragpipe.pipeline import PipelineDAG, Node, NodeType

    dag = PipelineDAG()
    dag.add_node(Node("embed", NodeType.TRANSFORM, fn=embedder.embed))
    dag.add_node(Node("retrieve_dense", NodeType.RETRIEVE, fn=dense_retriever.search))
    dag.add_node(Node("retrieve_sparse", NodeType.RETRIEVE, fn=sparse_retriever.search))
    dag.add_node(Node("merge", NodeType.MERGE, fn=rrf_merge))
    dag.add_node(Node("generate", NodeType.GENERATE, fn=generator.generate))

    dag.add_edge("embed", "retrieve_dense")
    dag.add_edge("embed", "retrieve_sparse")  # parallel branch
    dag.add_edge("retrieve_dense", "merge")
    dag.add_edge("retrieve_sparse", "merge")
    dag.add_edge("merge", "generate")

    result = dag.execute({"query": "What is RAG?"})
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class NodeType(str, Enum):
    """Types of nodes in the pipeline DAG."""
    INPUT = "input"
    TRANSFORM = "transform"
    RETRIEVE = "retrieve"
    MERGE = "merge"
    CONDITION = "condition"
    GENERATE = "generate"
    OUTPUT = "output"


@dataclass
class Node:
    """A node in the pipeline DAG."""
    name: str
    node_type: NodeType = NodeType.TRANSFORM
    fn: Callable | None = None
    condition_fn: Callable | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def execute(self, inputs: dict[str, Any]) -> Any:
        """Execute this node's function with the given inputs."""
        if self.fn is None:
            return inputs
        return self.fn(inputs)


@dataclass
class Edge:
    """A directed edge between two nodes."""
    source: str
    target: str
    condition: Callable | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeResult:
    """Result of executing a single node."""
    node_name: str
    output: Any = None
    duration_ms: float = 0.0
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node_name,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
        }


@dataclass
class DAGResult:
    """Result of executing the full pipeline DAG."""
    output: Any = None
    node_results: list[NodeResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    nodes_executed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"DAG Execution: {self.nodes_executed} nodes in {self.total_duration_ms:.1f}ms",
        ]
        for nr in self.node_results:
            status = "✓" if nr.status == "ok" else "✗"
            lines.append(f"  {status} {nr.node_name}: {nr.duration_ms:.1f}ms")
            if nr.error:
                lines.append(f"      error: {nr.error}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_duration_ms": round(self.total_duration_ms, 2),
            "nodes_executed": self.nodes_executed,
            "node_results": [nr.to_dict() for nr in self.node_results],
        }


class PipelineDAG:
    """Directed Acyclic Graph for complex RAG pipeline orchestration.

    Supports:
    - Sequential steps
    - Parallel branches (fan-out)
    - Merge points (fan-in)
    - Conditional routing
    - Error handling per node
    """

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._adjacency: dict[str, list[str]] = {}
        self._reverse_adj: dict[str, list[str]] = {}

    def add_node(self, node: Node) -> PipelineDAG:
        """Add a node to the DAG."""
        self._nodes[node.name] = node
        if node.name not in self._adjacency:
            self._adjacency[node.name] = []
        if node.name not in self._reverse_adj:
            self._reverse_adj[node.name] = []
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Callable | None = None,
    ) -> PipelineDAG:
        """Add a directed edge from source to target."""
        edge = Edge(source=source, target=target, condition=condition)
        self._edges.append(edge)
        if source not in self._adjacency:
            self._adjacency[source] = []
        self._adjacency[source].append(target)
        if target not in self._reverse_adj:
            self._reverse_adj[target] = []
        self._reverse_adj[target].append(source)
        return self

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm for topological sorting."""
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        for edge in self._edges:
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            order.append(current)
            for neighbor in self._adjacency.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._nodes):
            raise ValueError("Pipeline DAG contains a cycle")

        return order

    def _get_source_nodes(self) -> list[str]:
        """Get nodes with no incoming edges (entry points)."""
        return [
            name for name in self._nodes
            if not self._reverse_adj.get(name)
        ]

    def _get_sink_nodes(self) -> list[str]:
        """Get nodes with no outgoing edges (exit points)."""
        return [
            name for name in self._nodes
            if not self._adjacency.get(name)
        ]

    def execute(self, initial_input: dict[str, Any] | None = None) -> DAGResult:
        """Execute the DAG in topological order.

        Each node receives the outputs of its parent nodes as input.
        For merge nodes (multiple parents), inputs are combined into a dict.
        """
        t0 = time.perf_counter()
        order = self._topological_sort()
        node_outputs: dict[str, Any] = {}
        node_results: list[NodeResult] = []

        for node_name in order:
            node = self._nodes[node_name]

            # Gather inputs from parent nodes
            parents = self._reverse_adj.get(node_name, [])
            if not parents:
                # Source node: use initial input
                node_input = initial_input or {}
            elif len(parents) == 1:
                # Single parent: pass output directly
                node_input = node_outputs.get(parents[0], {})
            else:
                # Multiple parents (merge): combine outputs
                node_input = {
                    parent: node_outputs.get(parent)
                    for parent in parents
                }

            # Check edge conditions
            skip = False
            for edge in self._edges:
                if edge.target == node_name and edge.condition:
                    parent_output = node_outputs.get(edge.source)
                    if not edge.condition(parent_output):
                        skip = True
                        break

            # Check node condition
            if node.condition_fn and not node.condition_fn(node_input):
                skip = True

            if skip:
                node_outputs[node_name] = node_input
                node_results.append(NodeResult(
                    node_name=node_name, status="skipped",
                ))
                continue

            # Execute node
            nt0 = time.perf_counter()
            try:
                output = node.execute(node_input)
                duration = (time.perf_counter() - nt0) * 1000
                node_outputs[node_name] = output
                node_results.append(NodeResult(
                    node_name=node_name,
                    output=output,
                    duration_ms=duration,
                ))
            except Exception as e:
                duration = (time.perf_counter() - nt0) * 1000
                node_outputs[node_name] = None
                node_results.append(NodeResult(
                    node_name=node_name,
                    duration_ms=duration,
                    status="error",
                    error=str(e),
                ))

        # Final output is from sink nodes
        sinks = self._get_sink_nodes()
        if len(sinks) == 1:
            final_output = node_outputs.get(sinks[0])
        else:
            final_output = {s: node_outputs.get(s) for s in sinks}

        total_duration = (time.perf_counter() - t0) * 1000
        return DAGResult(
            output=final_output,
            node_results=node_results,
            total_duration_ms=total_duration,
            nodes_executed=len([r for r in node_results if r.status != "skipped"]),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def get_node(self, name: str) -> Node | None:
        return self._nodes.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {"name": n.name, "type": n.node_type.value}
                for n in self._nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target}
                for e in self._edges
            ],
        }

    def summary(self) -> str:
        order = self._topological_sort()
        lines = [f"PipelineDAG: {self.node_count} nodes, {self.edge_count} edges"]
        lines.append(f"  Execution order: {' → '.join(order)}")
        for name in order:
            node = self._nodes[name]
            parents = self._reverse_adj.get(name, [])
            children = self._adjacency.get(name, [])
            lines.append(
                f"  [{node.node_type.value}] {name}: "
                f"in={parents or 'start'} out={children or 'end'}"
            )
        return "\n".join(lines)
