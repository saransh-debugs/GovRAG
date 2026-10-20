"""Presidio in worker processes.

Why processes and not `await` or a thread:
  Presidio runs spaCy NER: ~80 ms of pure-Python/Cython CPU per paragraph, holding the GIL.
  Called on the event loop it freezes every other request for that long. In a thread it
  still holds the GIL most of the time, so the loop stutters. A separate process has its
  own GIL; the loop just awaits a future.

Two lanes (separate pools):
  ingest - bulk documents, 2 workers by default.
  query  - one short question per request.
  A 10 MB ingest must not make every user's question wait behind it, so they never share
  workers (a bulkhead).

Memory: each worker loads its own spaCy model (~700 MB RSS for en_core_web_lg). Worker
count is bounded by RAM, not cores. spaCy's string store also grows with every new word it
sees, so workers are recycled after `max_tasks_per_child` tasks to cap that growth.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from govrag.errors import Overloaded, Unavailable
from govrag.pii.entities import DEFAULT_MIN_SCORE, MIN_SCORE

Span = tuple[int, int, str, float]  # start, end, entity_type, score
Lane = Literal["ingest", "query"]
T = TypeVar("T")

_ENGINE: Any = None
# Area 000/666/9xx, group 00 and serial 0000 are never issued, so they're not SSNs.
SSN_DASHED = r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
# 3-3-4 digits. Presidio's phone recognizer validates area codes, so "276-018-9555" is
# "not a phone" and passes through raw. For data protection, phone-shaped is enough.
PHONE_SHAPED = r"(?<![\w-])(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![\w-])"
# spaCy reliably misses titled names in some contexts ("Contact Dr. Brett Robinson today"),
# measured in eval/pii_recall.py. A title plus capitalized words is a name.
TITLED_NAME = (r"\b(?:Dr|Mr|Mrs|Ms|Mx|Prof)\.?\s+[A-Z][a-z'\-]+"
               r"(?:\s+[A-Z][a-z'\-]+){0,2}\b")


def _init_worker(model: str) -> None:
    # Imported here so the parent process never loads spaCy (saves ~700 MB there).
    import logging

    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    global _ENGINE
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model}],
        }
    )
    _ENGINE = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    # Presidio's own SSN recognizer drops "known fake" numbers like 078-05-1120, and the
    # phone recognizer then claims them as PHONE_NUMBER. The digits still get hidden, but
    # with the wrong type, a role allowed to unmask phones would see an SSN. So: anything
    # shaped like a dashed SSN is an SSN, scored above the phone match.
    _ENGINE.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_SSN",
            name="DashedSsnRecognizer",
            patterns=[Pattern("ssn_dashed", SSN_DASHED, 0.9)],
        )
    )
    _ENGINE.registry.add_recognizer(PatternRecognizer(
        supported_entity="PHONE_NUMBER", name="PhoneShapeRecognizer",
        patterns=[Pattern("phone_shaped", PHONE_SHAPED, 0.5)],
    ))
    _ENGINE.registry.add_recognizer(PatternRecognizer(
        supported_entity="PERSON", name="TitledNameRecognizer",
        patterns=[Pattern("titled_name", TITLED_NAME, 0.7)],
    ))
    _ENGINE.analyze("warm up John Smith", language="en")


def resolve_overlaps(spans: Sequence[Span]) -> list[Span]:
    """Keep the best non-overlapping spans.

    Presidio returns overlaps, e.g. URL "acme.com" inside EMAIL "j@acme.com". Replacing
    both would corrupt the text. Higher score wins; on a tie, the longer span wins.
    """
    chosen: list[Span] = []
    for s in sorted(spans, key=lambda s: (-s[3], -(s[1] - s[0]), s[0])):
        if all(s[1] <= c[0] or s[0] >= c[1] for c in chosen):
            chosen.append(s)
    return sorted(chosen)


def _analyze_batch(texts: list[str], entities: list[str]) -> list[list[Span]]:
    assert _ENGINE is not None, "worker not initialized"
    out: list[list[Span]] = []
    for t in texts:
        if not t.strip():
            out.append([])
            continue
        res = _ENGINE.analyze(t, language="en", entities=entities)
        spans = [
            (r.start, r.end, r.entity_type, float(r.score))
            for r in res
            if r.score >= MIN_SCORE.get(r.entity_type, DEFAULT_MIN_SCORE) and r.end > r.start
        ]
        out.append(resolve_overlaps(spans))
    return out


@dataclass
class _LanePool:
    name: Lane
    workers: int
    model: str
    max_tasks_per_child: int
    max_pending: int
    executor: ProcessPoolExecutor = field(init=False)
    pending: int = 0

    def __post_init__(self) -> None:
        self.executor = self._make()

    def _make(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self.model,),
            max_tasks_per_child=self.max_tasks_per_child,
        )

    def rebuild(self) -> None:
        # A worker died (e.g. OOM-killed). The executor is now permanently broken and every
        # future submit would fail, so swap in a fresh one.
        old = self.executor
        self.executor = self._make()
        old.shutdown(wait=False, cancel_futures=True)


class PiiAnalyzer:
    def __init__(
        self,
        *,
        model: str,
        ingest_workers: int,
        query_workers: int,
        max_tasks_per_child: int = 2000,
        batch_chars: int = 20_000,
    ) -> None:
        self._lanes: dict[Lane, _LanePool] = {
            "ingest": _LanePool("ingest", ingest_workers, model, max_tasks_per_child,
                                max_pending=ingest_workers * 8),
            "query": _LanePool("query", query_workers, model, max_tasks_per_child,
                               max_pending=query_workers * 16),
        }
        self._batch_chars = batch_chars

    async def start(self) -> None:
        """Load spaCy in every worker now, not on the first user request (~9 s each)."""
        await asyncio.gather(
            *(
                asyncio.get_running_loop().run_in_executor(
                    lane.executor, _analyze_batch, ["warm"], ["PERSON"]
                )
                for lane in self._lanes.values()
                for _ in range(lane.workers)
            )
        )

    def close(self) -> None:
        for lane in self._lanes.values():
            lane.executor.shutdown(wait=False, cancel_futures=True)

    async def run(self, lane: Lane, fn: Callable[..., T], *args: Any, timeout_s: float) -> T:
        """Run any picklable CPU-bound function on a lane's workers (see ingest_cpu.py)."""
        return await self._submit_fn(self._lanes[lane], fn, args, timeout_s)

    async def _submit(
        self, lane: _LanePool, texts: list[str], entities: list[str], timeout_s: float
    ) -> list[list[Span]]:
        return await self._submit_fn(lane, _analyze_batch, (texts, entities), timeout_s)

    async def _submit_fn(
        self, lane: _LanePool, fn: Callable[..., T], args: tuple[Any, ...], timeout_s: float
    ) -> T:
        # Bounded queue: past max_pending, shed load instead of letting futures (and their
        # pickled text) pile up in memory.
        if lane.pending >= lane.max_pending:
            raise Overloaded(f"pii_{lane.name}_queue_full", retry_after_s=1.0)
        lane.pending += 1
        try:
            fut = asyncio.get_running_loop().run_in_executor(lane.executor, fn, *args)
            # Note: timing out frees this request, not the worker. A running task in a
            # process can't be interrupted. Input size caps (max_section_chars) bound how
            # long any one task can run.
            return await asyncio.wait_for(fut, timeout_s)
        except TimeoutError as e:
            raise Unavailable(f"pii_{lane.name}_timeout") from e
        except BrokenProcessPool as e:
            lane.rebuild()
            raise Unavailable(f"pii_{lane.name}_worker_died") from e
        finally:
            lane.pending -= 1

    async def analyze(
        self,
        texts: Sequence[str],
        entities: frozenset[str],
        *,
        lane: Lane,
        timeout_s: float = 60.0,
    ) -> list[list[Span]]:
        """Spans for each text, same order. Batches small texts to cut IPC overhead and
        spreads big inputs across workers."""
        ents = sorted(entities)
        pool = self._lanes[lane]
        batches: list[list[int]] = [[]]
        size = 0
        for i, t in enumerate(texts):
            if batches[-1] and size + len(t) > self._batch_chars:
                batches.append([])
                size = 0
            batches[-1].append(i)
            size += len(t)
        results = await asyncio.gather(
            *(self._submit(pool, [texts[i] for i in b], ents, timeout_s) for b in batches if b)
        )
        out: list[list[Span]] = [[] for _ in texts]
        for b, res in zip((b for b in batches if b), results, strict=True):
            for i, spans in zip(b, res, strict=True):
                out[i] = spans
        return out
