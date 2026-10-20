"""A stuck Presidio task must not keep its lane busy after the request times out."""

from __future__ import annotations

import time

import pytest

from govrag.config import Settings
from govrag.errors import Unavailable
from govrag.pii.analyzer import PiiAnalyzer
from govrag.pii.entities import DEFAULT_REDACT
from tests.helpers import sleep_then


async def test_timeout_kills_the_stuck_worker(settings: Settings) -> None:
    a = PiiAnalyzer(model=settings.spacy_model, ingest_workers=1, query_workers=1)
    await a.start()
    try:
        with pytest.raises(Unavailable, match="pii_ingest_timeout"):
            await a.run("ingest", sleep_then, 45.0, timeout_s=0.5)
        # Without the kill, this would queue behind a 45 s sleep.
        t0 = time.perf_counter()
        spans = await a.analyze(["Margaret Okafor joined the team."], DEFAULT_REDACT,
                                lane="ingest", timeout_s=60)
        assert spans[0] and time.perf_counter() - t0 < 30
    finally:
        a.close()


async def test_one_big_call_is_not_shed_by_its_own_fan_out(settings: Settings) -> None:
    # batch_chars=10 forces one batch per text: 60 batches vs a pending limit of 8.
    a = PiiAnalyzer(model=settings.spacy_model, ingest_workers=1, query_workers=1,
                    batch_chars=10)
    await a.start()
    try:
        texts = [f"Section {i}: Margaret Okafor approved it." for i in range(60)]
        spans = await a.analyze(texts, DEFAULT_REDACT, lane="ingest", timeout_s=120)
        assert len(spans) == 60 and all(spans)
    finally:
        a.close()
