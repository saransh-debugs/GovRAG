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
from govrag.pii.tokens import TOKEN_RE

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


def rescan_redact(
    texts: list[str], spans: list[list[Span]], hmac_key: bytes
) -> tuple[list[str], dict[str, Found]]:
    """Re-redact already-redacted chunk texts for newly added PII types.

    Spans touching an existing token are dropped: spaCy happily tags
    "PERSON_3f9a1c2e4b7d9a01" as an ORGANIZATION, and redacting that would nest tokens.
    """
    clean: list[list[Span]] = []
    for text, sp in zip(texts, spans, strict=True):
        taken = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        clean.append([s for s in sp if all(s[1] <= a or s[0] >= b for a, b in taken)])
    clean = propagate(texts, clean)
    # Same filter after propagation. Word boundaries already stop a propagated value from
    # landing inside a token; this makes it impossible rather than unlikely.
    final: list[list[Span]] = []
    for text, sp in zip(texts, clean, strict=True):
        taken = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        final.append([s for s in sp if all(s[1] <= a or s[0] >= b for a, b in taken)])
    red = [redact(t, sp, hmac_key) for t, sp in zip(texts, final, strict=True)]
    found: dict[str, Found] = {}
    for r in red:
        found.update(r.found)
    return [r.text for r in red], found
