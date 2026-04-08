"""Tests for ragpipe.pipeline.dag — Pipeline DAG."""

import pytest
from ragpipe.pipeline.dag import PipelineDAG, Node, Edge, NodeType, DAGResult


# ── Node ──────────────────────────────────────────────────────────────────────

def test_node_execute_with_fn():
    node = Node("double", fn=lambda x: x * 2)
    assert node.execute(5) == 10


def test_node_execute_passthrough():
    node = Node("noop")
    assert node.execute({"key": "val"}) == {"key": "val"}


# ── DAGResult ─────────────────────────────────────────────────────────────────

def test_dag_result_summary():
    r = DAGResult(total_duration_ms=10.5, nodes_executed=3)
    s = r.summary()
    assert "3 nodes" in s


def test_dag_result_to_dict():
    r = DAGResult(total_duration_ms=5.0, nodes_executed=2)
    d = r.to_dict()
    assert d["nodes_executed"] == 2


# ── PipelineDAG basics ───────────────────────────────────────────────────────

def test_dag_empty():
    dag = PipelineDAG()
    assert dag.node_count == 0
    assert dag.edge_count == 0


def test_dag_add_node():
    dag = PipelineDAG()
    dag.add_node(Node("a"))
    assert dag.node_count == 1


def test_dag_add_edge():
    dag = PipelineDAG()
    dag.add_node(Node("a"))
    dag.add_node(Node("b"))
    dag.add_edge("a", "b")
    assert dag.edge_count == 1


def test_dag_chaining():
    dag = PipelineDAG()
    result = dag.add_node(Node("a"))
    assert result is dag


# ── Linear execution ─────────────────────────────────────────────────────────

def test_dag_linear_execution():
    dag = PipelineDAG()
    dag.add_node(Node("input", NodeType.INPUT, fn=lambda x: {**x, "step": 1}))
    dag.add_node(Node("process", NodeType.TRANSFORM, fn=lambda x: {**x, "step": 2}))
    dag.add_node(Node("output", NodeType.OUTPUT, fn=lambda x: {**x, "step": 3}))
    dag.add_edge("input", "process")
    dag.add_edge("process", "output")

    result = dag.execute({"data": "test"})
    assert result.nodes_executed == 3
    assert result.output["step"] == 3


def test_dag_single_node():
    dag = PipelineDAG()
    dag.add_node(Node("only", fn=lambda x: "result"))
    result = dag.execute({"input": True})
    assert result.output == "result"
    assert result.nodes_executed == 1


# ── Parallel branches (fan-out / fan-in) ─────────────────────────────────────

def test_dag_parallel_branches():
    dag = PipelineDAG()
    dag.add_node(Node("start", fn=lambda x: x))
    dag.add_node(Node("branch_a", fn=lambda x: {"a": True}))
    dag.add_node(Node("branch_b", fn=lambda x: {"b": True}))
    dag.add_node(Node("merge", fn=lambda x: {"merged": True, **{k: v for d in x.values() if isinstance(d, dict) for k, v in d.items()}}))

    dag.add_edge("start", "branch_a")
    dag.add_edge("start", "branch_b")
    dag.add_edge("branch_a", "merge")
    dag.add_edge("branch_b", "merge")

    result = dag.execute({"query": "test"})
    assert result.nodes_executed == 4
    assert result.output is not None


# ── Conditional execution ────────────────────────────────────────────────────

def test_dag_conditional_node():
    dag = PipelineDAG()
    dag.add_node(Node("start", fn=lambda x: x))
    dag.add_node(Node("conditional", fn=lambda x: "executed",
                       condition_fn=lambda x: x.get("run_me", False)))
    dag.add_edge("start", "conditional")

    # Condition is False → node skipped
    result = dag.execute({"run_me": False})
    skipped = [r for r in result.node_results if r.status == "skipped"]
    assert len(skipped) == 1


def test_dag_conditional_node_passes():
    dag = PipelineDAG()
    dag.add_node(Node("start", fn=lambda x: {"run_me": True}))
    dag.add_node(Node("conditional", fn=lambda x: "executed",
                       condition_fn=lambda x: x.get("run_me", False)))
    dag.add_edge("start", "conditional")

    result = dag.execute({})
    assert result.nodes_executed == 2


# ── Edge conditions ──────────────────────────────────────────────────────────

def test_dag_edge_condition():
    dag = PipelineDAG()
    dag.add_node(Node("start", fn=lambda x: {"score": 0.3}))
    dag.add_node(Node("high_quality", fn=lambda x: "good"))
    dag.add_edge("start", "high_quality", condition=lambda output: output.get("score", 0) > 0.5)

    result = dag.execute({})
    # high_quality should be skipped because score < 0.5
    skipped = [r for r in result.node_results if r.status == "skipped"]
    assert len(skipped) == 1


# ── Error handling ───────────────────────────────────────────────────────────

def test_dag_node_error():
    dag = PipelineDAG()
    dag.add_node(Node("bad", fn=lambda x: 1/0))
    result = dag.execute({})
    assert result.node_results[0].status == "error"
    assert "division by zero" in result.node_results[0].error


# ── Cycle detection ──────────────────────────────────────────────────────────

def test_dag_cycle_detection():
    dag = PipelineDAG()
    dag.add_node(Node("a"))
    dag.add_node(Node("b"))
    dag.add_edge("a", "b")
    dag.add_edge("b", "a")

    with pytest.raises(ValueError, match="cycle"):
        dag.execute({})


# ── Topological sort ─────────────────────────────────────────────────────────

def test_dag_topological_order():
    dag = PipelineDAG()
    dag.add_node(Node("c"))
    dag.add_node(Node("a"))
    dag.add_node(Node("b"))
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    order = dag._topological_sort()
    assert order.index("a") < order.index("b") < order.index("c")


# ── Properties / serialization ───────────────────────────────────────────────

def test_dag_get_node():
    dag = PipelineDAG()
    dag.add_node(Node("test", NodeType.TRANSFORM))
    node = dag.get_node("test")
    assert node is not None
    assert node.node_type == NodeType.TRANSFORM


def test_dag_get_node_not_found():
    dag = PipelineDAG()
    assert dag.get_node("nonexistent") is None


def test_dag_to_dict():
    dag = PipelineDAG()
    dag.add_node(Node("a"))
    dag.add_node(Node("b"))
    dag.add_edge("a", "b")
    d = dag.to_dict()
    assert len(d["nodes"]) == 2
    assert len(d["edges"]) == 1


def test_dag_summary():
    dag = PipelineDAG()
    dag.add_node(Node("a", NodeType.INPUT))
    dag.add_node(Node("b", NodeType.OUTPUT))
    dag.add_edge("a", "b")
    s = dag.summary()
    assert "PipelineDAG" in s
    assert "a" in s
    assert "b" in s
