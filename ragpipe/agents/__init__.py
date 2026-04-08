"""Agentic RAG — intelligent query routing, self-correction, and multi-step retrieval."""

from ragpipe.agents.router import QueryRouter, RouteDecision, RouteType
from ragpipe.agents.crag import CRAGAgent, CRAGResult, CRAGAction, RelevanceGrade
from ragpipe.agents.adaptive import AdaptiveRetriever, AdaptiveResult, QueryComplexity
from ragpipe.agents.selfrag import SelfRAGAgent, SelfRAGResult, SelfRAGReflection
from ragpipe.agents.react import ReActAgent, ReActResult, ReActStep, Tool
from ragpipe.agents.smart_pipeline import SmartPipeline, SmartResult

__all__ = [
    "QueryRouter", "RouteDecision", "RouteType",
    "CRAGAgent", "CRAGResult", "CRAGAction", "RelevanceGrade",
    "AdaptiveRetriever", "AdaptiveResult", "QueryComplexity",
    "SelfRAGAgent", "SelfRAGResult", "SelfRAGReflection",
    "ReActAgent", "ReActResult", "ReActStep", "Tool",
    "SmartPipeline", "SmartResult",
]
