"""Rehydration: allowlist, aliases, bare tokens, and streaming split at every position."""

from __future__ import annotations

import os
import re

from govrag.pii.tokens import bracket, token_for
from govrag.rehydrate import MAX_TOKEN_LEN, Rehydrator

K = os.urandom(32)
P1 = token_for(K, "PERSON", "Margaret Okafor")[0]
P2 = token_for(K, "PERSON", "Daniel Whitfield")[0]
SSN = token_for(K, "US_SSN", "078-05-1120")[0]
HEX16 = re.compile(r"[0-9a-f]{16}")


def test_reveals_only_allowlisted_and_aliases_the_rest() -> None:
    rh = Rehydrator({P1: "Margaret Okafor"})
    out = rh.whole(f"{bracket(P1)} approved {bracket(P2)}'s raise; SSN {bracket(SSN)}.")
    assert out == "Margaret Okafor approved [PERSON 1]'s raise; SSN [US_SSN 1]."
    assert rh.unmasked_types == {"PERSON"} and rh.masked_types == {"PERSON", "US_SSN"}
    assert not HEX16.search(out)


def test_aliases_are_stable_within_an_answer_and_numbered_per_answer() -> None:
    rh = Rehydrator({})
    out = rh.whole(f"{bracket(P2)} met {bracket(P1)}. Later {bracket(P2)} left.")
    assert out == "[PERSON 1] met [PERSON 2]. Later [PERSON 1] left."
    # A new answer starts its numbering over, so aliases can't link answers.
    assert Rehydrator({}).whole(bracket(P1)) == "[PERSON 1]"


def test_bare_tokens_are_handled_too() -> None:
    out = Rehydrator({}).whole(f"The approver was {P1}, see doc.")
    assert out == "The approver was [PERSON 1], see doc."


def test_non_tokens_untouched() -> None:
    text = "See [doc 2] and [1]; ACME_corp and [NOTATOKEN_xyz] stay."
    assert Rehydrator({}).whole(text) == text


def test_stream_equals_whole_at_every_split() -> None:
    text = (f"Hi {bracket(P1)}, {bracket(P2)} and {P1} (bare). SSN: {bracket(SSN)}. "
            "[doc 1] END")
    want = Rehydrator({P1: "Margaret Okafor"}).whole(text)
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1, 7):
            rh = Rehydrator({P1: "Margaret Okafor"})
            got = rh.feed(text[:i]) + rh.feed(text[i:j]) + rh.feed(text[j:]) + rh.flush()
            assert got == want, (i, j)


def test_stream_char_by_char() -> None:
    text = f"x {bracket(P1)} y {P2} z"
    rh = Rehydrator({P1: "M"})
    out = "".join(rh.feed(c) for c in text) + rh.flush()
    assert out == "x M y [PERSON 1] z"


def test_stray_bracket_does_not_stall_the_stream() -> None:
    rh = Rehydrator({})
    emitted = rh.feed("[" + "a" * (MAX_TOKEN_LEN + 5))
    assert emitted.startswith("[a")  # released once it can't be a token any more
