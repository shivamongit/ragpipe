"""Tests for ragpipe.agents.react — reasoning + acting agent loop."""

import pytest

from ragpipe.agents.react import ReActAgent, ReActResult, ReActStep, Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _search_tool(query):
    return "Paris is the capital of France."


def _calc_tool(expression):
    return str(eval(expression))  # noqa: S307 — test-only


async def _async_search(query):
    return "Paris is the capital of France (async)."


def _make_tools():
    return [
        Tool(name="search", description="Search the knowledge base", fn=_search_tool),
        Tool(name="calculate", description="Evaluate a math expression", fn=_calc_tool),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tool_creation():
    t = Tool(name="search", description="Search the web", fn=_search_tool)
    assert t.name == "search"
    assert t.description == "Search the web"
    assert t.fn is _search_tool
    assert t.async_fn is None


def test_react_heuristic_fallback():
    """Without reason_fn, the agent runs each tool once with the query."""
    agent = ReActAgent(tools=_make_tools())
    result = agent.query("What is the capital of France?")
    assert isinstance(result, ReActResult)
    assert len(result.steps) == 2  # one per tool
    assert "search" in result.tools_used
    assert "calculate" in result.tools_used
    assert result.metadata.get("heuristic") is True


def test_react_with_llm():
    """Mock LLM that returns a Final Answer on the first step."""
    def mock_reason(prompt):
        return (
            "Thought: I know the answer already.\n"
            "Action: Final Answer\n"
            "Action Input: Paris is the capital of France."
        )

    agent = ReActAgent(reason_fn=mock_reason, tools=_make_tools())
    result = agent.query("What is the capital of France?")
    assert result.answer == "Paris is the capital of France."
    assert result.total_steps == 1


def test_react_multi_step():
    """LLM uses search tool first, then gives final answer."""
    step_count = {"n": 0}

    def mock_reason(prompt):
        step_count["n"] += 1
        if step_count["n"] == 1:
            return (
                "Thought: I need to search for this.\n"
                "Action: search\n"
                "Action Input: capital of France"
            )
        return (
            "Thought: I now have enough information.\n"
            "Action: Final Answer\n"
            "Action Input: Paris is the capital of France."
        )

    agent = ReActAgent(reason_fn=mock_reason, tools=_make_tools())
    result = agent.query("What is the capital of France?")
    assert result.total_steps == 2
    assert "search" in result.tools_used
    assert result.answer == "Paris is the capital of France."


def test_react_max_steps_limit():
    """Agent should stop after max_steps even if no Final Answer."""
    def never_finish(prompt):
        return (
            "Thought: Let me search again.\n"
            "Action: search\n"
            "Action Input: capital of France"
        )

    agent = ReActAgent(reason_fn=never_finish, tools=_make_tools(), max_steps=3)
    result = agent.query("What is the capital of France?")
    assert result.total_steps == 3
    assert result.metadata.get("max_steps_reached") is True


def test_react_final_answer_detection():
    assert ReActAgent._is_final_answer("Final Answer") is True
    assert ReActAgent._is_final_answer("final answer") is True
    assert ReActAgent._is_final_answer("search") is False


def test_react_tool_execution():
    agent = ReActAgent(tools=_make_tools())
    obs = agent._execute_tool("search", "test query")
    assert obs == "Paris is the capital of France."


def test_react_unknown_tool():
    agent = ReActAgent(tools=_make_tools())
    obs = agent._execute_tool("nonexistent", "test")
    assert "Error" in obs
    assert "Unknown tool" in obs


def test_react_empty_tools():
    agent = ReActAgent()
    result = agent.query("anything")
    assert isinstance(result, ReActResult)
    assert result.steps == []
    assert result.answer == "No tools produced useful results."


def test_react_parse_response():
    response = (
        "Thought: I should search for this.\n"
        "Action: search\n"
        "Action Input: capital of France"
    )
    thought, action, action_input = ReActAgent._parse_response(response)
    assert "search" in thought.lower() or thought != ""
    assert action == "search"
    assert action_input == "capital of France"


def test_react_parse_response_fallback():
    """Unparseable response is treated as Final Answer."""
    thought, action, action_input = ReActAgent._parse_response("Just a plain string.")
    assert action == "Final Answer"


def test_react_result_fields():
    result = ReActResult(
        answer="test answer",
        steps=[],
        total_steps=0,
        tools_used=["search"],
        confidence=0.5,
        metadata={"key": "value"},
    )
    assert result.answer == "test answer"
    assert result.total_steps == 0
    assert result.tools_used == ["search"]
    assert result.confidence == 0.5
    assert result.metadata["key"] == "value"


def test_react_confidence_computation():
    steps = [
        ReActStep(thought="t", action="search", action_input="q",
                  observation="Paris is capital", step_number=1),
        ReActStep(thought="t", action="Final Answer", action_input="Paris",
                  observation="", step_number=2),
    ]
    conf = ReActAgent._compute_confidence(steps)
    assert 0.0 <= conf <= 1.0
    assert conf > 0  # has informative observation + final answer


@pytest.mark.asyncio
async def test_react_async():
    tools = [
        Tool(name="search", description="Search", fn=_search_tool,
             async_fn=_async_search),
    ]

    def mock_reason(prompt):
        return (
            "Thought: I know the answer.\n"
            "Action: Final Answer\n"
            "Action Input: Paris."
        )

    agent = ReActAgent(reason_fn=mock_reason, tools=tools)
    result = await agent.aquery("What is the capital of France?")
    assert isinstance(result, ReActResult)
    assert result.answer == "Paris."
