"""ReAct Agent — Reasoning + Acting loop with tool use.

Implements the ReAct paradigm (Yao et al., 2022) where the agent alternates between
reasoning about the current situation and taking actions using external tools:

1. **Thought**: Reason about what to do next given the question and prior observations
2. **Action**: Choose a tool and provide input
3. **Observation**: Process the tool's output
4. Repeat until the agent produces a **Final Answer**

This module is framework-agnostic — supply any tools as plain callables and an
optional LLM as the reasoning engine.  When no LLM is provided, a heuristic
fallback executes every registered tool once and returns the combined observations.

Usage:
    from ragpipe.agents import ReActAgent, Tool

    agent = ReActAgent(
        reason_fn=my_llm,
        tools=[
            Tool(name="search", description="Search knowledge base", fn=my_search),
            Tool(name="calculate", description="Do math", fn=my_calc),
        ],
    )
    result = agent.query("What is 15% of the company revenue mentioned in the report?")
    print(result.answer, result.steps, result.tools_used)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """A tool that the ReAct agent can use."""
    name: str
    description: str
    fn: Callable
    async_fn: Optional[Callable] = None


@dataclass
class ReActStep:
    """A single step in the ReAct reasoning chain."""
    thought: str
    action: str
    action_input: str
    observation: str
    step_number: int


@dataclass
class ReActResult:
    """Result from ReAct agent execution."""
    answer: str
    steps: list[ReActStep] = field(default_factory=list)
    total_steps: int = 0
    tools_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Prompt template ──────────────────────────────────────────────────────────

REACT_PROMPT = """You are a reasoning agent that solves problems step by step using tools.

Available tools:
{tool_descriptions}

Use the following format EXACTLY:

Thought: <your reasoning about what to do next>
Action: <tool name OR "Final Answer">
Action Input: <input to the tool, or the final answer text>

If you have enough information to answer, use:
Thought: I now have enough information.
Action: Final Answer
Action Input: <your complete answer>

Question: {question}

{scratchpad}"""

_FINAL_ANSWER = "final answer"


class ReActAgent:
    """ReAct Agent — Reasoning + Acting loop with tool use.

    Alternates between reasoning (Thought), acting (Action + Action Input),
    and observing (Observation) until the agent emits a ``Final Answer`` action
    or the step limit is reached.
    """

    def __init__(
        self,
        reason_fn: Optional[Callable] = None,
        tools: Optional[list[Tool]] = None,
        max_steps: int = 5,
        verbose: bool = False,
    ):
        self.reason_fn = reason_fn
        self.tools: dict[str, Tool] = {}
        for tool in (tools or []):
            self.tools[tool.name.lower()] = tool
        self.max_steps = max_steps
        self.verbose = verbose

    # ── Public API ───────────────────────────────────────────────────────

    def query(self, question: str, **kwargs: Any) -> ReActResult:
        """Execute the ReAct loop: Thought → Action → Observation → repeat."""
        if self.reason_fn is None:
            return self._heuristic_react(question, **kwargs)

        steps: list[ReActStep] = []
        tools_used: list[str] = []

        for step_num in range(1, self.max_steps + 1):
            prompt = self._build_prompt(question, steps)
            raw_response = self.reason_fn(prompt)
            thought, action, action_input = self._parse_response(raw_response)

            if self._is_final_answer(action):
                step = ReActStep(
                    thought=thought,
                    action="Final Answer",
                    action_input=action_input,
                    observation="",
                    step_number=step_num,
                )
                steps.append(step)
                return ReActResult(
                    answer=action_input,
                    steps=steps,
                    total_steps=step_num,
                    tools_used=tools_used,
                    confidence=self._compute_confidence(steps),
                )

            # Execute the chosen tool
            observation = self._execute_tool(action, action_input)
            if action.lower() not in [t.lower() for t in tools_used]:
                tools_used.append(action)

            step = ReActStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                step_number=step_num,
            )
            steps.append(step)

        # Max steps reached — synthesise a best-effort answer
        answer = self._synthesise_fallback(question, steps)
        return ReActResult(
            answer=answer,
            steps=steps,
            total_steps=self.max_steps,
            tools_used=tools_used,
            confidence=self._compute_confidence(steps) * 0.7,
            metadata={"max_steps_reached": True},
        )

    async def aquery(self, question: str, **kwargs: Any) -> ReActResult:
        """Async version of query."""
        if self.reason_fn is None:
            return await asyncio.to_thread(self._heuristic_react, question, **kwargs)

        steps: list[ReActStep] = []
        tools_used: list[str] = []

        for step_num in range(1, self.max_steps + 1):
            prompt = self._build_prompt(question, steps)
            raw_response = await asyncio.to_thread(self.reason_fn, prompt)
            thought, action, action_input = self._parse_response(raw_response)

            if self._is_final_answer(action):
                step = ReActStep(
                    thought=thought,
                    action="Final Answer",
                    action_input=action_input,
                    observation="",
                    step_number=step_num,
                )
                steps.append(step)
                return ReActResult(
                    answer=action_input,
                    steps=steps,
                    total_steps=step_num,
                    tools_used=tools_used,
                    confidence=self._compute_confidence(steps),
                )

            observation = await self._aexecute_tool(action, action_input)
            if action.lower() not in [t.lower() for t in tools_used]:
                tools_used.append(action)

            step = ReActStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                step_number=step_num,
            )
            steps.append(step)

        answer = self._synthesise_fallback(question, steps)
        return ReActResult(
            answer=answer,
            steps=steps,
            total_steps=self.max_steps,
            tools_used=tools_used,
            confidence=self._compute_confidence(steps) * 0.7,
            metadata={"max_steps_reached": True},
        )

    # ── Prompt building ──────────────────────────────────────────────────

    def _build_prompt(
        self, question: str, steps: list[ReActStep],
    ) -> str:
        """Build the ReAct prompt with tool descriptions and scratchpad."""
        tool_lines = []
        for tool in self.tools.values():
            tool_lines.append(f"- {tool.name}: {tool.description}")
        tool_descriptions = "\n".join(tool_lines) if tool_lines else "(no tools available)"

        # Build scratchpad from prior steps
        scratchpad_lines: list[str] = []
        for step in steps:
            scratchpad_lines.append(f"Thought: {step.thought}")
            scratchpad_lines.append(f"Action: {step.action}")
            scratchpad_lines.append(f"Action Input: {step.action_input}")
            scratchpad_lines.append(f"Observation: {step.observation}")
        scratchpad = "\n".join(scratchpad_lines)

        return REACT_PROMPT.format(
            tool_descriptions=tool_descriptions,
            question=question,
            scratchpad=scratchpad,
        )

    # ── Response parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str, str]:
        """Extract Thought, Action, and Action Input from LLM response."""
        thought = ""
        action = ""
        action_input = ""

        # Try structured regex patterns
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\Z)", response, re.DOTALL,
        )
        action_match = re.search(
            r"Action:\s*(.+?)(?=\nAction Input:|\Z)", response, re.DOTALL,
        )
        input_match = re.search(
            r"Action Input:\s*(.+?)(?=\nThought:|\nObservation:|\Z)",
            response, re.DOTALL,
        )

        if thought_match:
            thought = thought_match.group(1).strip()
        if action_match:
            action = action_match.group(1).strip()
        if input_match:
            action_input = input_match.group(1).strip()

        # Fallback: if nothing parsed, treat the whole response as a final answer
        if not action:
            action = "Final Answer"
            action_input = response.strip()
            thought = "Could not parse structured response; treating as final answer."

        return thought, action, action_input

    # ── Tool execution ───────────────────────────────────────────────────

    def _execute_tool(self, action: str, action_input: str) -> str:
        """Execute a tool by name and return its observation."""
        tool = self.tools.get(action.lower())
        if tool is None:
            return f"Error: Unknown tool '{action}'. Available: {', '.join(self.tools)}"
        try:
            result = tool.fn(action_input)
            return str(result) if result is not None else "(no output)"
        except Exception as exc:
            return f"Error executing {action}: {exc}"

    async def _aexecute_tool(self, action: str, action_input: str) -> str:
        """Async tool execution — prefers async_fn when available."""
        tool = self.tools.get(action.lower())
        if tool is None:
            return f"Error: Unknown tool '{action}'. Available: {', '.join(self.tools)}"
        try:
            if tool.async_fn is not None:
                result = await tool.async_fn(action_input)
            else:
                result = await asyncio.to_thread(tool.fn, action_input)
            return str(result) if result is not None else "(no output)"
        except Exception as exc:
            return f"Error executing {action}: {exc}"

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_final_answer(action: str) -> bool:
        """Check whether the action signals a final answer."""
        return action.lower().strip() == _FINAL_ANSWER

    @staticmethod
    def _compute_confidence(steps: list[ReActStep]) -> float:
        """Derive confidence from the reasoning trace."""
        if not steps:
            return 0.0
        # More observations with content → higher confidence
        informative = sum(
            1 for s in steps
            if s.observation and not s.observation.startswith("Error")
        )
        # Shorter chains with informative results score higher
        base = min(informative / 3, 1.0) * 0.6
        # Final-answer step present
        if steps[-1].action.lower().strip() == _FINAL_ANSWER:
            base += 0.4
        return min(base, 1.0)

    @staticmethod
    def _synthesise_fallback(
        question: str, steps: list[ReActStep],
    ) -> str:
        """Build a best-effort answer from accumulated observations."""
        observations = [
            s.observation for s in steps
            if s.observation and not s.observation.startswith("Error")
        ]
        if observations:
            return "Based on available information: " + " ".join(observations)
        return "I was unable to find a satisfactory answer within the step limit."

    # ── Heuristic fallback ───────────────────────────────────────────────

    def _heuristic_react(self, question: str, **kwargs: Any) -> ReActResult:
        """Execute each tool once in order when no LLM is available."""
        steps: list[ReActStep] = []
        tools_used: list[str] = []
        observations: list[str] = []

        for idx, (name, tool) in enumerate(self.tools.items(), start=1):
            observation = self._execute_tool(name, question)
            tools_used.append(name)
            step = ReActStep(
                thought=f"Trying tool '{name}' with the original question.",
                action=name,
                action_input=question,
                observation=observation,
                step_number=idx,
            )
            steps.append(step)
            if observation and not observation.startswith("Error"):
                observations.append(observation)

        if observations:
            answer = "Based on available information: " + " ".join(observations)
            confidence = min(len(observations) / max(len(self.tools), 1), 1.0) * 0.5
        else:
            answer = "No tools produced useful results."
            confidence = 0.0

        return ReActResult(
            answer=answer,
            steps=steps,
            total_steps=len(steps),
            tools_used=tools_used,
            confidence=confidence,
            metadata={"heuristic": True},
        )
