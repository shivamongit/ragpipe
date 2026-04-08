"""OpenTelemetry export for ragpipe traces.

Converts ragpipe Tracer spans to OpenTelemetry format and exports them
via OTLP (gRPC or HTTP), Jaeger, or console.

Usage (with OpenTelemetry SDK installed):
    from ragpipe.observability import Tracer
    from ragpipe.observability.otel import OTelExporter

    exporter = OTelExporter(service_name="my-rag-app")
    tracer = Tracer()

    # ... run pipeline with tracer ...

    exporter.export(tracer)

Usage (without OpenTelemetry SDK — console/JSON fallback):
    exporter = OTelExporter(service_name="my-app", backend="console")
    exporter.export(tracer)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ragpipe.observability.tracer import Span, Tracer

logger = logging.getLogger(__name__)


@dataclass
class OTelExporter:
    """Export ragpipe traces to OpenTelemetry-compatible backends.

    Backends:
        - "otlp_http": Export via OTLP HTTP (requires opentelemetry-exporter-otlp-proto-http)
        - "otlp_grpc": Export via OTLP gRPC (requires opentelemetry-exporter-otlp-proto-grpc)
        - "console": Print spans to stdout (no dependencies)
        - "json": Return spans as JSON-serializable dicts
    """
    service_name: str = "ragpipe"
    backend: str = "console"
    endpoint: str = "http://localhost:4318"
    headers: dict[str, str] = field(default_factory=dict)
    resource_attributes: dict[str, str] = field(default_factory=dict)

    def export(self, tracer: Tracer) -> list[dict[str, Any]]:
        """Export all spans from a Tracer instance.

        Returns list of exported span dicts regardless of backend.
        """
        otel_spans = [self._convert_span(s) for s in tracer.spans]

        if self.backend == "console":
            self._export_console(otel_spans)
        elif self.backend == "json":
            pass  # Just return the dicts
        elif self.backend in ("otlp_http", "otlp_grpc"):
            self._export_otlp(otel_spans)
        else:
            logger.warning("Unknown backend '%s', falling back to console", self.backend)
            self._export_console(otel_spans)

        return otel_spans

    def _convert_span(self, span: Span) -> dict[str, Any]:
        """Convert a ragpipe Span to an OpenTelemetry-compatible dict."""
        # OTel uses nanosecond timestamps
        start_ns = int(span.start_time * 1_000_000_000) if span.start_time else 0
        end_ns = int(span.end_time * 1_000_000_000) if span.end_time else 0

        attributes = {
            "ragpipe.component": span.name,
            "ragpipe.duration_ms": round(span.duration_ms, 2),
        }
        for k, v in span.metadata.items():
            attr_key = f"ragpipe.{k}"
            # OTel attributes must be str, int, float, or bool
            if isinstance(v, (str, int, float, bool)):
                attributes[attr_key] = v
            else:
                attributes[attr_key] = str(v)

        otel_span = {
            "name": span.name,
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id or "",
            "start_time_unix_nano": start_ns,
            "end_time_unix_nano": end_ns,
            "duration_ms": round(span.duration_ms, 2),
            "status": {
                "code": "ERROR" if span.status == "error" else "OK",
                "message": span.error if span.status == "error" else "",
            },
            "attributes": attributes,
            "resource": {
                "service.name": self.service_name,
                **self.resource_attributes,
            },
        }

        return otel_span

    def _export_console(self, spans: list[dict[str, Any]]) -> None:
        """Print spans to stdout in a readable format."""
        for s in spans:
            print(
                f"[{s['resource']['service.name']}] "
                f"{s['name']} "
                f"trace={s['trace_id']} "
                f"span={s['span_id']} "
                f"duration={s['duration_ms']:.2f}ms "
                f"status={s['status']['code']}"
            )
            for k, v in s["attributes"].items():
                if not k.startswith("ragpipe.component") and not k.startswith("ragpipe.duration"):
                    print(f"  {k}={v}")

    def _export_otlp(self, spans: list[dict[str, Any]]) -> None:
        """Export spans via OpenTelemetry OTLP protocol."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
        except ImportError:
            logger.error(
                "OpenTelemetry SDK not installed. "
                "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
            )
            self._export_console(spans)
            return

        resource = Resource.create({
            "service.name": self.service_name,
            **self.resource_attributes,
        })

        provider = TracerProvider(resource=resource)

        if self.backend == "otlp_http":
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(
                    endpoint=f"{self.endpoint}/v1/traces",
                    headers=self.headers,
                )
            except ImportError:
                logger.error("Install: pip install opentelemetry-exporter-otlp-proto-http")
                return
        elif self.backend == "otlp_grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(
                    endpoint=self.endpoint,
                    headers=self.headers,
                )
            except ImportError:
                logger.error("Install: pip install opentelemetry-exporter-otlp-proto-grpc")
                return
        else:
            return

        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        otel_tracer = trace.get_tracer(self.service_name)

        for s in spans:
            with otel_tracer.start_as_current_span(s["name"]) as otel_span:
                for k, v in s["attributes"].items():
                    otel_span.set_attribute(k, v)
                if s["status"]["code"] == "ERROR":
                    otel_span.set_status(trace.StatusCode.ERROR, s["status"]["message"])

        provider.shutdown()

    def to_otlp_json(self, tracer: Tracer) -> str:
        """Export tracer spans as OTLP-compatible JSON string."""
        spans = [self._convert_span(s) for s in tracer.spans]
        payload = {
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": self.service_name}},
                        *[
                            {"key": k, "value": {"stringValue": str(v)}}
                            for k, v in self.resource_attributes.items()
                        ],
                    ],
                },
                "scopeSpans": [{
                    "scope": {"name": "ragpipe", "version": "3.0.0"},
                    "spans": [
                        {
                            "traceId": s["trace_id"],
                            "spanId": s["span_id"],
                            "parentSpanId": s.get("parent_span_id", ""),
                            "name": s["name"],
                            "startTimeUnixNano": str(s["start_time_unix_nano"]),
                            "endTimeUnixNano": str(s["end_time_unix_nano"]),
                            "attributes": [
                                {"key": k, "value": {"stringValue": str(v)}}
                                for k, v in s["attributes"].items()
                            ],
                            "status": {
                                "code": 2 if s["status"]["code"] == "ERROR" else 1,
                                "message": s["status"].get("message", ""),
                            },
                        }
                        for s in spans
                    ],
                }],
            }],
        }
        return json.dumps(payload, indent=2)
