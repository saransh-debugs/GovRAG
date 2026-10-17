"""Spans and logs never carry text: no PII values, no chunk text, no questions."""

from __future__ import annotations

import json
import logging
import uuid

import asyncpg
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from govrag.auth import Principal
from govrag.errors import Unavailable
from govrag.ingest import DocumentIn, SectionIn
from govrag.tracing import AllowlistExporter
from tests.security.test_query_security import PII, QUESTIONS, World, world  # noqa: F401

TEXT_FRAGMENTS = ["Compensation review", "12% raise", "offsite is in March", "Board memo",
                  "organizing", "Vacation", *QUESTIONS]


def dump(spans: tuple[ReadableSpan, ...] | list[ReadableSpan]) -> str:
    return "\n".join(json.dumps({
        "name": s.name, "attrs": dict(s.attributes or {}),
        "events": [(e.name, dict(e.attributes or {})) for e in s.events],
        "status": s.status.description,
    }, default=str) for s in spans)


def test_exporter_strips_everything_not_allowlisted() -> None:
    mem = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(AllowlistExporter(mem)))
    with tp.get_tracer("t").start_as_current_span("x") as sp:
        sp.set_attribute("govrag.tenant_id", "acme")
        sp.set_attribute("db.statement", "SELECT * FROM x WHERE name = 'Margaret Okafor'")
        sp.set_attribute("http.url", "/v1/query?q=078-05-1120")
        sp.record_exception(ValueError("SSN 078-05-1120 rejected"))
        sp.set_status(Status(StatusCode.ERROR, "078-05-1120"))
    (out,) = mem.get_finished_spans()
    assert dict(out.attributes or {}) == {"govrag.tenant_id": "acme"}
    assert [(e.name, dict(e.attributes or {})) for e in out.events] == [
        ("exception", {"exception.type": "ValueError"})]
    assert out.status.status_code == StatusCode.ERROR and not out.status.description
    assert "078-05-1120" not in dump([out]) and "Okafor" not in dump([out])


async def test_no_text_in_spans_or_logs_end_to_end(
    world: World, spans: InMemorySpanExporter, owner: asyncpg.Connection,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    spans.clear()
    await world.svc.ingestor.ingest(Principal(world.t, "svc", frozenset({"ingest"})), DocumentIn(
        "Vacation memo", ("hr",), (SectionIn(
            f"Vacation: {PII['margaret']} (SSN {PII['ssn_hr']}) takes leave."),)))
    for roles in (("hr",), ("intern",), ("exec",)):
        for q in QUESTIONS:
            await world.svc.query.answer(world.who(*roles), q)
    # Error paths too: exceptions are where text most often sneaks into telemetry.
    await owner.execute("REVOKE INSERT ON audit.events FROM govrag_app")
    try:
        with pytest.raises(Unavailable):
            await world.svc.query.answer(world.who("hr"), QUESTIONS[0])
    finally:
        await owner.execute("GRANT INSERT ON audit.events TO govrag_app")
    world.box.models["claude-haiku-5.5"].fail = "midstream"
    try:
        with pytest.raises(Unavailable):
            await world.svc.query.answer(world.who("hr"), "compensation review raise")
    finally:
        world.box.reset()

    exported = dump(spans.get_finished_spans())
    assert len(spans.get_finished_spans()) > 100
    logs = caplog.text
    for v in [*PII.values(), "Okafor", "Whitfield", "Raghunathan", *TEXT_FRAGMENTS]:
        assert v not in exported, f"in spans: {v!r}"
        assert v not in logs, f"in logs: {v!r}"


async def test_spans_cover_every_stage(world: World, spans: InMemorySpanExporter) -> None:  # noqa: F811
    spans.clear()
    rid = uuid.uuid4()
    ans = await world.svc.query.answer(world.who("hr"), "compensation review raise",
                                       request_id=rid)
    got = spans.get_finished_spans()
    root = next(s for s in got if s.name == "query"
                and (s.attributes or {}).get("govrag.request_id") == str(rid))
    tree = [s for s in got if s.context.trace_id == root.context.trace_id]
    assert {s.name for s in tree} >= {
        "query", "policy", "redact_question", "embed", "retrieve", "audit.retrieve",
        "llm.first_token", "vault.wait", "audit.rehydrate",
    }
    a = root.attributes or {}
    assert a["govrag.tenant_id"] == world.t and a["govrag.model"] == ans.meta.model
    assert a["govrag.outcome"] == "ok" and "PERSON" in a["govrag.pii_types_unmasked"]
