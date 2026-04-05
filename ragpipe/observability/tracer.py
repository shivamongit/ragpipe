"""Pipeline tracer — structured tracing for every pipeline step.

Records timing, token counts, chunk IDs, and metadata for each step
in a query lifecycle. Outputs as JSON for debugging, dashboards, or
OpenTelemetry-compatible export.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Span:
    """A single traced operation within a pipeline execution."""
    name: str
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok, error
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
            "status": self.status,
            "error": self.error,
        }


class Tracer:
    """Trace pipeline execution with structured spans.

    Usage:
        tracer = Tracer()

        with tracer.span("embed") as s:
            embeddings = embedder.embed(texts)
            s.metadata["count"] = len(texts)

        with tracer.span("retrieve") as s:
            results = retriever.search(embedding, top_k=5)
            s.metadata["results"] = len(results)

        print(tracer.to_json())
        print(f"Total: {tracer.total_duration_ms:.1f}ms")
    """

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self._active_span: Optional[Span] = None

    def span(self, name: str, **metadata) -> _SpanContext:
        """Create a new span context manager."""
        return _SpanContext(self, name, metadata)

    def add_span(self, span: Span) -> None:
        """Add a completed span."""
        self.spans.append(span)

    @property
    def total_duration_ms(self) -> float:
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans)
        return (end - start) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Human-readable summary of the trace."""
        lines = [f"Trace {self.trace_id} ({self.total_duration_ms:.1f}ms total, {len(self.spans)} spans):"]
        for s in self.spans:
            status = "✓" if s.status == "ok" else "✗"
            lines.append(f"  {status} {s.name}: {s.duration_ms:.1f}ms")
            if s.metadata:
                for k, v in s.metadata.items():
                    lines.append(f"      {k}: {v}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all spans."""
        self.spans.clear()


class _SpanContext:
    """Context manager for tracing a span."""

    def __init__(self, tracer: Tracer, name: str, metadata: dict[str, Any]):
        self.tracer = tracer
        self.span = Span(
            name=name,
            trace_id=tracer.trace_id,
            span_id=uuid.uuid4().hex[:12],
            metadata=metadata,
        )

    def __enter__(self) -> Span:
        self.span.start_time = time.perf_counter()
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.span.end_time = time.perf_counter()
        self.span.duration_ms = (self.span.end_time - self.span.start_time) * 1000
        if exc_type:
            self.span.status = "error"
            self.span.error = str(exc_val)
        self.tracer.add_span(self.span)
        return False  # Don't suppress exceptions


class TracerCallback:
    """Callback interface for pipeline observability.

    Attach to a Pipeline to automatically trace all operations.

    Usage:
        callback = TracerCallback()
        # After pipeline query:
        callback.on_embed_start(texts)
        callback.on_embed_end(embeddings)
        print(callback.tracer.summary())
    """

    def __init__(self, tracer: Tracer | None = None):
        self.tracer = tracer or Tracer()
        self._current_span: Optional[Span] = None

    def on_query_start(self, question: str) -> None:
        self.tracer = Tracer()
        self._start_span("query", question=question)

    def on_embed_start(self, texts: list[str]) -> None:
        self._start_span("embed", text_count=len(texts))

    def on_embed_end(self, embeddings: list[list[float]]) -> None:
        self._end_span(embedding_count=len(embeddings), dim=len(embeddings[0]) if embeddings else 0)

    def on_retrieve_start(self, top_k: int) -> None:
        self._start_span("retrieve", top_k=top_k)

    def on_retrieve_end(self, results: list) -> None:
        self._end_span(
            result_count=len(results),
            top_score=results[0].score if results else 0,
        )

    def on_rerank_start(self, count: int) -> None:
        self._start_span("rerank", input_count=count)

    def on_rerank_end(self, results: list) -> None:
        self._end_span(output_count=len(results))

    def on_generate_start(self, question: str, context_count: int) -> None:
        self._start_span("generate", context_chunks=context_count)

    def on_generate_end(self, tokens_used: int, model: str) -> None:
        self._end_span(tokens_used=tokens_used, model=model)

    def on_query_end(self, latency_ms: float) -> None:
        if self._current_span:
            self._end_span(total_latency_ms=latency_ms)

    def _start_span(self, name: str, **metadata) -> None:
        self._current_span = Span(
            name=name,
            trace_id=self.tracer.trace_id,
            span_id=uuid.uuid4().hex[:12],
            start_time=time.perf_counter(),
            metadata=metadata,
        )

    def _end_span(self, **metadata) -> None:
        if self._current_span:
            self._current_span.end_time = time.perf_counter()
            self._current_span.duration_ms = (
                self._current_span.end_time - self._current_span.start_time
            ) * 1000
            self._current_span.metadata.update(metadata)
            self.tracer.add_span(self._current_span)
            self._current_span = None
