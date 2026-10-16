"""Replace PII spans with tokens. Pure functions, no I/O."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from govrag.pii.analyzer import Span, resolve_overlaps
from govrag.pii.tokens import bracket, random_token, token_for

MIN_PROPAGATE_CHARS = 4


def propagate(texts: Sequence[str], spans: Sequence[Sequence[Span]]) -> list[list[Span]]:
    """If a value is found once anywhere in a document, hide every exact copy of it.

    spaCy's name finding depends on the surrounding words: "Priyanka Raghunathan joined the
    team" is found, "Contact Priyanka Raghunathan today" is not. Names usually appear more
    than once in a document, so one good detection protects the bad-context mentions too.
    For PERSON, each capitalized part of 4+ letters also counts ("Raghunathan said...").

    Copies found this way get a slightly lower score, so a real detection that overlaps
    them still wins in resolve_overlaps.
    """
    known: dict[str, tuple[str, float]] = {}
    for text, sp in zip(texts, spans, strict=True):
        for start, end, etype, score in sp:
            value = text[start:end].strip()
            if len(value) >= MIN_PROPAGATE_CHARS:
                known.setdefault(value, (etype, score * 0.99))
            if etype == "PERSON":
                for part in value.split():
                    part = part.strip(".,;:'\"()")
                    if len(part) >= MIN_PROPAGATE_CHARS and part[0].isupper():
                        known.setdefault(part, ("PERSON", score * 0.95))
    if not known:
        return [list(sp) for sp in spans]
    # Longest first, so "Priyanka Raghunathan" wins over "Raghunathan" at the same spot.
    alts = "|".join(re.escape(v) for v in sorted(known, key=len, reverse=True))
    pattern = re.compile(rf"(?<!\w)(?:{alts})(?!\w)")
    out: list[list[Span]] = []
    for text, sp in zip(texts, spans, strict=True):
        extra = [
            (m.start(), m.end(), *known[m.group(0)]) for m in pattern.finditer(text)
        ]
        out.append(resolve_overlaps([*sp, *extra]) if extra else list(sp))
    return out


@dataclass(frozen=True, slots=True)
class Found:
    token: str
    entity_type: str
    value: str
    mac: bytes


@dataclass(slots=True)
class Redacted:
    text: str
    found: dict[str, Found] = field(default_factory=dict)  # token -> original

    @property
    def types(self) -> frozenset[str]:
        return frozenset(f.entity_type for f in self.found.values())


def redact(text: str, spans: Sequence[Span], hmac_key: bytes) -> Redacted:
    """Spans must be non-overlapping (analyzer.resolve_overlaps guarantees it)."""
    out: list[str] = []
    found: dict[str, Found] = {}
    pos = 0
    for start, end, etype, _ in sorted(spans):
        if start < pos:
            raise ValueError("overlapping spans")
        value = text[start:end]
        tok, mac = token_for(hmac_key, etype, value)
        found.setdefault(tok, Found(tok, etype, value, mac))
        out.append(text[pos:start])
        out.append(bracket(tok))
        pos = end
    out.append(text[pos:])
    return Redacted("".join(out), found)


@dataclass(slots=True)
class RedactedQuestion:
    text: str
    # Every token we put in the question -> what the user typed. The user already knows
    # these values, so putting them back in the answer reveals nothing.
    own: dict[str, str] = field(default_factory=dict)
    types: frozenset[str] = frozenset()


def redact_question(
    text: str, spans: Sequence[Span], hmac_key: bytes, may_unmask: frozenset[str]
) -> RedactedQuestion:
    """Like redact(), but policy-aware.

    If the asker may see this PII type: deterministic token, so "John Smith" in the
    question matches [PERSON_x] in the chunks. If not: a random token that matches
    nothing. Otherwise someone without PERSON access could guess names one by one and
    learn which ones appear in documents from whether search matches.
    """
    out: list[str] = []
    own: dict[str, str] = {}
    types: set[str] = set()
    pos = 0
    for start, end, etype, _ in sorted(spans):
        if start < pos:
            raise ValueError("overlapping spans")
        value = text[start:end]
        tok = token_for(hmac_key, etype, value)[0] if etype in may_unmask else random_token(etype)
        own[tok] = value
        types.add(etype)
        out.append(text[pos:start])
        out.append(bracket(tok))
        pos = end
    out.append(text[pos:])
    return RedactedQuestion("".join(out), own, frozenset(types))
