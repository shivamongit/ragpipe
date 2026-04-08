"""Tests for ragpipe.observability.otel — OpenTelemetry export."""

import json
from ragpipe.observability.tracer import Tracer
from ragpipe.observability.otel import OTelExporter


def _make_tracer_with_spans():
    tracer = Tracer(trace_id="test123")
    with tracer.span("embed", text_count=5):
        pass
    with tracer.span("retrieve", top_k=10) as s:
        s.metadata["result_count"] = 3
    return tracer


# ── OTelExporter basics ──────────────────────────────────────────────────────

def test_exporter_defaults():
    exp = OTelExporter()
    assert exp.service_name == "ragpipe"
    assert exp.backend == "console"


def test_exporter_json_backend():
    exp = OTelExporter(backend="json")
    tracer = _make_tracer_with_spans()
    spans = exp.export(tracer)
    assert len(spans) == 2
    assert spans[0]["name"] == "embed"
    assert spans[1]["name"] == "retrieve"


def test_exporter_convert_span():
    exp = OTelExporter(service_name="my-app")
    tracer = _make_tracer_with_spans()
    otel_span = exp._convert_span(tracer.spans[0])
    assert otel_span["name"] == "embed"
    assert otel_span["trace_id"] == "test123"
    assert otel_span["resource"]["service.name"] == "my-app"
    assert "ragpipe.text_count" in otel_span["attributes"]


def test_exporter_span_status_ok():
    exp = OTelExporter(backend="json")
    tracer = Tracer()
    with tracer.span("ok_step"):
        pass
    spans = exp.export(tracer)
    assert spans[0]["status"]["code"] == "OK"


def test_exporter_span_status_error():
    exp = OTelExporter(backend="json")
    tracer = Tracer()
    try:
        with tracer.span("bad_step"):
            raise ValueError("oops")
    except ValueError:
        pass
    spans = exp.export(tracer)
    assert spans[0]["status"]["code"] == "ERROR"
    assert "oops" in spans[0]["status"]["message"]


def test_exporter_attributes():
    exp = OTelExporter(backend="json")
    tracer = Tracer()
    with tracer.span("step", count=42, label="test"):
        pass
    spans = exp.export(tracer)
    attrs = spans[0]["attributes"]
    assert attrs["ragpipe.count"] == 42
    assert attrs["ragpipe.label"] == "test"


def test_exporter_resource_attributes():
    exp = OTelExporter(
        service_name="test-svc",
        resource_attributes={"env": "staging"},
        backend="json",
    )
    tracer = Tracer()
    with tracer.span("x"):
        pass
    spans = exp.export(tracer)
    assert spans[0]["resource"]["service.name"] == "test-svc"
    assert spans[0]["resource"]["env"] == "staging"


# ── Console export ────────────────────────────────────────────────────────────

def test_exporter_console_no_crash(capsys):
    exp = OTelExporter(backend="console")
    tracer = _make_tracer_with_spans()
    spans = exp.export(tracer)
    assert len(spans) == 2
    captured = capsys.readouterr()
    assert "embed" in captured.out
    assert "retrieve" in captured.out


def test_exporter_unknown_backend_falls_back(capsys):
    exp = OTelExporter(backend="unknown_backend")
    tracer = Tracer()
    with tracer.span("test"):
        pass
    spans = exp.export(tracer)
    assert len(spans) == 1
    captured = capsys.readouterr()
    assert "test" in captured.out


# ── OTLP JSON export ─────────────────────────────────────────────────────────

def test_to_otlp_json():
    exp = OTelExporter(service_name="my-rag")
    tracer = _make_tracer_with_spans()
    json_str = exp.to_otlp_json(tracer)
    data = json.loads(json_str)
    assert "resourceSpans" in data
    spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 2
    assert spans[0]["name"] == "embed"


def test_to_otlp_json_structure():
    exp = OTelExporter(service_name="test", resource_attributes={"version": "3.0"})
    tracer = Tracer()
    with tracer.span("step"):
        pass
    data = json.loads(exp.to_otlp_json(tracer))
    resource = data["resourceSpans"][0]["resource"]
    attr_keys = [a["key"] for a in resource["attributes"]]
    assert "service.name" in attr_keys
    assert "version" in attr_keys


# ── Empty tracer ──────────────────────────────────────────────────────────────

def test_export_empty_tracer():
    exp = OTelExporter(backend="json")
    tracer = Tracer()
    spans = exp.export(tracer)
    assert spans == []


def test_otlp_json_empty_tracer():
    exp = OTelExporter()
    tracer = Tracer()
    data = json.loads(exp.to_otlp_json(tracer))
    spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert spans == []
