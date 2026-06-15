"""OpenTelemetry tracing setup.

We use the vendor-neutral OpenTelemetry stack: OpenInference instrumentation
for LangChain (so every model/tool call is a span) exported to a local Arize
Phoenix UI. Because `register()` sets the *global* tracer provider, the manual
spans we open in `retrieval`/`verify`/`pipeline` (via `trace.get_tracer(...)`)
land in the same trace tree as the auto-instrumented LangChain spans.

Setup is best-effort: if Phoenix isn't installed or reachable, tracing degrades
to a no-op (or console, with GROUNDED_TRACE_CONSOLE=1) and never breaks a run.
"""

from __future__ import annotations

import os

from .config import Settings

_TRUTHY = {"1", "true", "yes", "on"}


def _console_tracing() -> None:
    """Fallback: emit spans to stdout so tracing is observable with no server."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def setup_tracing(settings: Settings) -> str | None:
    """Initialize tracing. Returns a human-readable destination, or None if off."""
    if not settings.tracing:
        return None

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    try:
        from phoenix.otel import register

        register(project_name="grounded", auto_instrument=True)
        return endpoint
    except Exception:
        if os.getenv("GROUNDED_TRACE_CONSOLE", "").strip().lower() in _TRUTHY:
            _console_tracing()
            return "console"
        return None
