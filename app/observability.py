"""Observability module for edgewalker-be (backend).

Mirrors ``shared/observability.py`` from edgewalker-runtime but lives
inside the backend's own package since the backend does not mount the
``shared/`` volume.

Initialises OpenTelemetry traces + metrics + structured logging.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

_OTEL_AVAILABLE = False

try:
    from opentelemetry import trace, context
    from opentelemetry.propagate import inject, extract
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore[import-untyped]

    _OTEL_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

_initialised = False
_service_name: str = "backend"


class _TraceContextFilter(logging.Filter):
    """Inject ``trace_id`` and ``span_id`` into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _OTEL_AVAILABLE:
            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.trace_id:
                record.trace_id = format(ctx.trace_id, "032x")  # type: ignore[attr-defined]
                record.span_id = format(ctx.span_id, "016x")  # type: ignore[attr-defined]
            else:
                record.trace_id = "0" * 32  # type: ignore[attr-defined]
                record.span_id = "0" * 16  # type: ignore[attr-defined]
        else:
            record.trace_id = "0" * 32  # type: ignore[attr-defined]
            record.span_id = "0" * 16  # type: ignore[attr-defined]
        return True


class JSONLogFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "service": _service_name,
            "trace_id": getattr(record, "trace_id", "0" * 32),
            "span_id": getattr(record, "span_id", "0" * 16),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class HealthCheckFilter(logging.Filter):
    """Suppress noisy health-check access-log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "GET /health" not in msg


def init_telemetry(
    service_name: str = "backend",
    *,
    enable_json_logs: bool = True,
    log_level: str | None = None,
) -> None:
    """Initialise OpenTelemetry traces + metrics + structured logging."""
    global _initialised, _service_name

    if _initialised:
        return
    _initialised = True
    _service_name = os.getenv("OTEL_SERVICE_NAME", service_name)

    level_name = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.addFilter(_TraceContextFilter())

    if enable_json_logs:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONLogFormatter())
        handler.addFilter(HealthCheckFilter())
        root.handlers = [handler]
    else:
        for h in root.handlers or [logging.StreamHandler()]:
            h.addFilter(HealthCheckFilter())

    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    # Suppress noisy third-party loggers
    for _noisy_logger in (
        "urllib3",
        "asyncio",
        "websockets",
        "hpack",
        "httpcore",
        "httpx",
        "grpc",
    ):
        logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint or not _OTEL_AVAILABLE:
        if not _OTEL_AVAILABLE:
            logger.info("OpenTelemetry SDK not installed — telemetry disabled")
        else:
            logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — telemetry disabled")
        return

    logger.info("Initialising OpenTelemetry  service=%s  endpoint=%s", _service_name, endpoint)

    resource = Resource.create({SERVICE_NAME: _service_name})

    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(span_exporter, max_queue_size=2048, max_export_batch_size=512, schedule_delay_millis=5000)
    )
    trace.set_tracer_provider(tracer_provider)

    metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    from opentelemetry import metrics as otel_metrics
    otel_metrics.set_meter_provider(meter_provider)

    # Suppress noisy SDK exporter logs when collector is unreachable
    for _exporter_logger_name in (
        "opentelemetry.exporter.otlp.proto.grpc.exporter",
        "opentelemetry.exporter.otlp.proto.grpc._log_exporter",
        "opentelemetry.sdk.trace.export",
        "opentelemetry.sdk.metrics.export",
        "opentelemetry.sdk._logs.export",
    ):
        logging.getLogger(_exporter_logger_name).setLevel(logging.CRITICAL)

    # Auto-instrumentation
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
    except ImportError:
        pass
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor().instrument()
    except ImportError:
        pass
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        SQLAlchemyInstrumentor().instrument()
    except ImportError:
        pass
    try:
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
        Psycopg2Instrumentor().instrument()
    except ImportError:
        pass

    logger.info("OpenTelemetry initialised  (traces + metrics)")


def instrument_app(app: Any) -> None:
    """Instrument a FastAPI application instance with OTel tracing."""
    if not _OTEL_AVAILABLE or not _initialised:
        return
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return
    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:
        logger.warning("Failed to instrument FastAPI app: %s", exc)


def get_tracer(name: str | None = None) -> Any:
    """Return an OTel Tracer."""
    if _OTEL_AVAILABLE:
        return trace.get_tracer(name or _service_name)

    class _NoOp:
        def start_as_current_span(self, name, **kw):
            return _NoOpSpan()
        def start_span(self, name, **kw):
            return _NoOpSpan()

    return _NoOp()


def get_meter(name: str | None = None) -> Any:
    """Return an OTel Meter."""
    if _OTEL_AVAILABLE:
        from opentelemetry import metrics as otel_metrics
        return otel_metrics.get_meter(name or _service_name)

    class _NoOp:
        def create_counter(self, *a, **kw): return _NoOpInstrument()
        def create_histogram(self, *a, **kw): return _NoOpInstrument()
        def create_up_down_counter(self, *a, **kw): return _NoOpInstrument()

    return _NoOp()


class _NoOpSpan:
    def set_attribute(self, key, value): ...
    def set_status(self, *a, **kw): ...
    def record_exception(self, exc): ...
    def __enter__(self): return self
    def __exit__(self, *a): ...


class _NoOpInstrument:
    def add(self, amount=1, attributes=None): ...
    def record(self, amount=0, attributes=None): ...
