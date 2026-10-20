"""Synthetic documents with every PII value labeled, for measuring detection recall.

Names appear in "easy" contexts (subject of a sentence) and "hard" ones (after a verb,
in lists, lowercase-adjacent), because that's where spaCy misses. Each document mentions
its people more than once, like real documents do.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from faker import Faker

Span = tuple[int, int, str]  # start, end, type


@dataclass(frozen=True)
class Doc:
    sections: list[str]
    gold: list[list[Span]]  # per section


EASY = [
    "{p} approved the budget for the next quarter.",
    "{p} joined the platform team in March.",
    "According to {p}, the migration is on schedule.",
]
HARD = [
    "Contact {p} today about the offsite.",
    "Ask {p} for the badge.",
    "cc: {p}, legal, finance",
    "the reviewer was {p} and nobody objected",
    "Thanks {p}!",
    "From: {p}",
    "Owner: {p} (finance)",
    "Acme: {p} gets a raise.",
]
STRUCTURED = [
    ("SSN on file: {v}.", "US_SSN", lambda f: f.ssn()),
    ("Reach them at {v} for follow-ups.", "EMAIL_ADDRESS", lambda f: f.email()),
    ("Phone: {v}", "PHONE_NUMBER", lambda f: f.numerify("###-###-####")),
    ("Card ending in full number {v} was charged.", "CREDIT_CARD",
     lambda f: f.credit_card_number(card_type="visa16")),
]


_TITLES = ("Dr. ", "Mr. ", "Mrs. ", "Ms. ", "Miss ", "Mx. ", "Prof. ")
_SUFFIXES = (" MD", " PhD", " DDS", " DVM", " Jr.", " Sr.", " II", " III", " IV", " V")


def _core(value: str) -> tuple[int, int]:
    """The part of a Faker name that is actually PII. "Dr." and "MD" aren't: leaving
    them visible next to a placeholder reveals a title, not a person."""
    start, end = 0, len(value)
    for t in _TITLES:
        if value.startswith(t):
            start = len(t)
    for suf in _SUFFIXES:
        if value[:end].endswith(suf):
            end -= len(suf)
    return start, end


def _fill(template: str, value: str, etype: str) -> tuple[str, Span]:
    i = template.index("{")
    text = template[:i] + value + template[template.index("}") + 1:]
    a, b = _core(value) if etype == "PERSON" else (0, len(value))
    return text, (i + a, i + b, etype)


def build(n_docs: int = 150, seed: int = 7) -> list[Doc]:
    fake = Faker("en_US")
    Faker.seed(seed)
    rng = random.Random(seed)
    docs: list[Doc] = []
    # Like a real company: half the people in a document are recurring employees, half are
    # one-offs. Recurring names are what cross-document lookup can catch.
    staff = [fake.name() for _ in range(60)]
    for _ in range(n_docs):
        people = [rng.choice(staff) if rng.random() < 0.5 else fake.name()
                  for _ in range(rng.randint(1, 3))]
        sections: list[str] = []
        gold: list[list[Span]] = []
        for _s in range(rng.randint(2, 4)):
            parts: list[str] = []
            spans: list[Span] = []
            pos = 0
            for _k in range(rng.randint(2, 4)):
                if rng.random() < 0.6:
                    tmpl = rng.choice(EASY if rng.random() < 0.5 else HARD)
                    text, sp = _fill(tmpl, rng.choice(people), "PERSON")
                else:
                    tmpl, etype, gen = rng.choice(STRUCTURED)
                    text, sp = _fill(tmpl, gen(fake), etype)
                spans.append((sp[0] + pos, sp[1] + pos, sp[2]))
                parts.append(text)
                pos += len(text) + 1
            sections.append(" ".join(parts))
            gold.append(spans)
        docs.append(Doc(sections, gold))
    return docs
