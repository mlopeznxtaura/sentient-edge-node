"""
telemetry/tracing.py — OpenTelemetry tracing (Cluster 01: Sentient Edge Node)

Instruments the agent loop with distributed tracing spans.
Exports to OTLP endpoint (Jaeger, Grafana Tempo, etc.).

SDKs: opentelemetry-sdk, opentelemetry-exporter-otlp
"""
import logging
import functools
from typing import Optional

logger = logging.getLogger(__name__)


class EdgeTracer:
    def __init__(self, config: dict):
        self._cfg = config.get("telemetry", {})
        self._tracer = None
        self._enabled = False

    def load(self) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": "sentient-edge-node"})
            provider = TracerProvider(resource=resource)

            otlp_endpoint = self._cfg.get("otlp_endpoint", "http://localhost:4317")
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("edge-node")
            self._enabled = True
            logger.info("OpenTelemetry tracing enabled -> %s", otlp_endpoint)
        except ImportError:
            logger.warning("opentelemetry not installed — tracing disabled")
        except Exception as e:
            logger.warning("Tracing setup failed: %s — disabled", e)

    def span(self, name: str):
        """Context manager for a trace span. No-op if tracing disabled."""
        if self._tracer:
            return self._tracer.start_as_current_span(name)
        return _NoOpSpan()

    def trace(self, fn):
        """Decorator to wrap a function in a span named after it."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self.span(fn.__qualname__):
                return fn(*args, **kwargs)
        return wrapper

    @property
    def enabled(self) -> bool:
        return self._enabled


class _NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, *_): pass
