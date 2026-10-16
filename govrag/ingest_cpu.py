"""The CPU-heavy middle of ingest, run inside an ingest worker process.

Measured on this repo: LlamaIndex's SentenceSplitter in a thread held the GIL for up to
235 ms at a time (it's mostly pure Python), freezing the event loop just as badly as
running it inline. The ingest workers already exist (they host spaCy), so propagation,
redaction and chunking run there too. The web process only does I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from govrag.pii.analyzer import Span
from govrag.pii.redact import Found, propagate, redact

_CHUNKER: Any = None


@dataclass(frozen=True, slots=True)
class Prepared:
    title: str
    chunks: list[list[str]]  # per section
    found: dict[str, Found]


def prepare_document(
    texts: list[str],
    spans: list[list[Span]],
    hmac_key: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> Prepared:
    """texts[0] is the title, the rest are sections."""
    global _CHUNKER
    if _CHUNKER is None:
        from govrag.chunking import Chunker

        _CHUNKER = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    spans = propagate(texts, spans)
    red = [redact(t, sp, hmac_key) for t, sp in zip(texts, spans, strict=True)]
    found: dict[str, Found] = {}
    for r in red:
        found.update(r.found)
    return Prepared(red[0].text, [_CHUNKER.split(r.text) for r in red[1:]], found)
