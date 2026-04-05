"""Tests for observability / tracing."""

import json
import time

from ragpipe.observability.tracer import Tracer, Span, TracerCallback


def test_tracer_span_basic():
    tracer = Tracer()
    with tracer.span("embed") as s:
        s.metadata["count"] = 5
        time.sleep(0.01)

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "embed"
    assert tracer.spans[0].duration_ms > 0
    assert tracer.spans[0].metadata["count"] == 5
    assert tracer.spans[0].status == "ok"


def test_tracer_multiple_spans():
    tracer = Tracer()
    with tracer.span("embed"):
        pass
    with tracer.span("retrieve"):
        pass
    with tracer.span("generate"):
        pass

    assert len(tracer.spans) == 3
    names = [s.name for s in tracer.spans]
    assert names == ["embed", "retrieve", "generate"]


def test_tracer_error_span():
    tracer = Tracer()
    try:
        with tracer.span("failing") as s:
            raise ValueError("something broke")
    except ValueError:
        pass

    assert len(tracer.spans) == 1
    assert tracer.spans[0].status == "error"
    assert "something broke" in tracer.spans[0].error


def test_tracer_total_duration():
    tracer = Tracer()
    with tracer.span("step1"):
        time.sleep(0.01)
    with tracer.span("step2"):
        time.sleep(0.01)

    assert tracer.total_duration_ms > 15  # at least ~20ms


def test_tracer_to_dict():
    tracer = Tracer(trace_id="test123")
    with tracer.span("embed"):
        pass

    d = tracer.to_dict()
    assert d["trace_id"] == "test123"
    assert d["span_count"] == 1
    assert len(d["spans"]) == 1


def test_tracer_to_json():
    tracer = Tracer()
    with tracer.span("embed"):
        pass

    j = tracer.to_json()
    data = json.loads(j)
    assert "spans" in data


def test_tracer_summary():
    tracer = Tracer()
    with tracer.span("embed") as s:
        s.metadata["dim"] = 768
    with tracer.span("retrieve"):
        pass

    summary = tracer.summary()
    assert "embed" in summary
    assert "retrieve" in summary
    assert "dim: 768" in summary


def test_tracer_clear():
    tracer = Tracer()
    with tracer.span("step"):
        pass
    tracer.clear()
    assert len(tracer.spans) == 0


def test_span_to_dict():
    span = Span(
        name="test",
        trace_id="t1",
        span_id="s1",
        start_time=1.0,
        end_time=2.0,
        duration_ms=1000.0,
        metadata={"key": "value"},
    )
    d = span.to_dict()
    assert d["name"] == "test"
    assert d["duration_ms"] == 1000.0
    assert d["metadata"]["key"] == "value"


def test_tracer_callback():
    cb = TracerCallback()
    cb.on_query_start("What is X?")
    cb.on_embed_start(["text1", "text2"])
    cb.on_embed_end([[0.1], [0.2]])
    cb.on_retrieve_start(5)

    assert len(cb.tracer.spans) >= 1
