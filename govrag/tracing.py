"""OpenTelemetry spans with an attribute allowlist enforced at export.

Spans are for debugging and latency, not compliance (that's audit.events: spans get
sampled and dropped under load). But spans still leave the process, so they must never
carry text. The guard sits in the exporter, not at the call sites:
  - attributes: only keys on ALLOWED survive. A library (or a future us) adding
    `db.statement` or `http.url` with a query string gets silently stripped.
  - events: only "exception", and only its exception.type. exception.message and the
    stacktrace can contain values (asyncpg puts row values in DETAIL, for example).
  - status descriptions: dropped, same reason.
Checking at the call sites alone would be one forgotten attribute away from a leak.

Async generators: a span made "current" across a `yield` breaks contextvars (the token
is detached in a different Context than it was attached in). The streaming query path
passes span contexts explicitly and only uses start_as_current_span around plain awaits.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, Status

ALLOWED: frozenset[str] = frozenset({
    "govrag.tenant_id", "govrag.user_id", "govrag.request_id",
    "govrag.document_ids", "govrag.chunk_ids", "govrag.hits", "govrag.k",
    "govrag.pii_types", "govrag.pii_types_unmasked", "govrag.pii_types_masked",
    "govrag.policy_version", "govrag.model", "govrag.rehydration", "govrag.lane",
    "govrag.outcome", "govrag.error_code", "govrag.chunks", "govrag.new_tokens",
    "govrag.audit.action",
    "http.route", "http.request.method", "http.response.status_code",
})

tracer = trace.get_tracer("govrag")


class AllowlistExporter(SpanExporter):
    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    @staticmethod
    def scrub(s: ReadableSpan) -> ReadableSpan:
        attrs = {k: v for k, v in (s.attributes or {}).items() if k in ALLOWED}
        events = [
            Event("exception", {"exception.type": str(e.attributes.get("exception.type", ""))},
                  e.timestamp)
            for e in s.events if e.name == "exception" and e.attributes
        ]
        return ReadableSpan(
            name=s.name, context=s.context, parent=s.parent, resource=s.resource,
            attributes=attrs, events=events, links=(), kind=s.kind,
            status=Status(s.status.status_code), start_time=s.start_time,
            end_time=s.end_time, instrumentation_scope=s.instrumentation_scope,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self.scrub(s) for s in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._inner.force_flush(timeout_millis)


def setup(endpoint: str, *, service: str = "govrag") -> TracerProvider | None:
    """Install a provider exporting to OTLP/HTTP. No endpoint = no export at all.

    BatchSpanProcessor, not Simple: Simple exports on the request path, so a slow
    collector would add its latency to every request. Batch drops spans when its queue is
    full, which is fine for spans and exactly why spans aren't the audit log.
    """
    if not endpoint or isinstance(trace.get_tracer_provider(), TracerProvider):
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(
        AllowlistExporter(OTLPSpanExporter(endpoint=endpoint, timeout=5)),
        max_queue_size=4096,
    ))
    trace.set_tracer_provider(provider)
    return provider


@contextmanager
def step(name: str, parent: Span | None = None, **attrs: Any) -> Iterator[Span]:
    """A child span around a block that contains no `yield`."""
    ctx = trace.set_span_in_context(parent) if parent is not None else otel_context.get_current()
    with tracer.start_as_current_span(name, context=ctx,
                                      attributes=_attrs(attrs)) as span:
        yield span


def _attrs(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        key = f"govrag.{k}"
        if isinstance(v, set | frozenset | list | tuple):
            out[key] = [str(x) for x in sorted(v, key=str)]
        elif isinstance(v, bool | int | float | str):
            out[key] = v
        else:
            out[key] = str(v)
    return out


def annotate(span: Span, **attrs: Any) -> None:
    span.set_attributes(_attrs(attrs))
