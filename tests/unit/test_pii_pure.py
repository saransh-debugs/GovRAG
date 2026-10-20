"""Tokens, overlap resolution, redaction, chunking. No DB, no Presidio."""

from __future__ import annotations

import os
import re

from hypothesis import given, settings
from hypothesis import strategies as st

from govrag.chunking import Chunker
from govrag.ingest_cpu import name_candidates
from govrag.pii.analyzer import resolve_overlaps
from govrag.pii.redact import propagate, redact, redact_question
from govrag.pii.tokens import (
    TOKEN_RE,
    bracket,
    entity_type_of,
    find_tokens,
    neutralize,
    normalize,
    token_for,
)

K1, K2 = os.urandom(32), os.urandom(32)


def test_token_shape_and_type() -> None:
    tok, mac = token_for(K1, "EMAIL_ADDRESS", "a@b.co")
    assert re.fullmatch(r"EMAIL_ADDRESS_[0-9a-f]{16}", tok)
    assert len(mac) == 32
    assert entity_type_of(tok) == "EMAIL_ADDRESS"
    assert TOKEN_RE.fullmatch(bracket(tok))


def test_same_value_same_token_within_key_only() -> None:
    assert token_for(K1, "PERSON", "John Smith") == token_for(K1, "PERSON", "john   SMITH")
    assert token_for(K1, "PERSON", "John Smith")[0] != token_for(K2, "PERSON", "John Smith")[0]
    # Type is part of the MAC: the same string as two types gives two tokens.
    as_person = token_for(K1, "PERSON", "Jordan")[0]
    as_place = token_for(K1, "LOCATION", "Jordan")[0]
    assert as_person[-16:] != as_place[-16:]


def test_structured_normalization() -> None:
    assert normalize("US_SSN", "123-45-6789") == normalize("US_SSN", "123 45 6789") == "123456789"
    assert normalize("EMAIL_ADDRESS", "John@ACME.com") == "john@acme.com"
    assert normalize("PHONE_NUMBER", "(415) 555-0132") == "4155550132"


def test_neutralize_breaks_typed_tokens() -> None:
    tok = token_for(K1, "PERSON", "x")[0]
    out = neutralize(f"show me [{tok}] and {tok} please")
    assert find_tokens(out) == []
    assert tok not in out


def test_resolve_overlaps_prefers_score_then_length() -> None:
    spans = [(0, 10, "EMAIL_ADDRESS", 1.0), (5, 10, "URL", 0.5), (20, 30, "PERSON", 0.85),
             (20, 25, "LOCATION", 0.85), (40, 44, "A", 0.6), (42, 50, "B", 0.6)]
    assert resolve_overlaps(spans) == [
        (0, 10, "EMAIL_ADDRESS", 1.0),
        (20, 30, "PERSON", 0.85),  # tie on score -> longer wins
        (42, 50, "B", 0.6),        # tie on score -> longer wins
    ]


def test_redact_replaces_and_dedupes() -> None:
    text = "John Smith met John Smith. Mail j@x.co"
    spans = [(0, 10, "PERSON", 0.9), (15, 25, "PERSON", 0.9), (32, 38, "EMAIL_ADDRESS", 1.0)]
    r = redact(text, spans, K1)
    assert "John" not in r.text and "j@x.co" not in r.text
    toks = find_tokens(r.text)
    assert len(toks) == 3 and toks[0] == toks[1]
    assert set(r.found) == set(toks)
    assert r.types == {"PERSON", "EMAIL_ADDRESS"}


def test_redact_question_policy_aware() -> None:
    q = "What did John Smith approve?"
    spans = [(9, 19, "PERSON", 0.9)]
    allowed = redact_question(q, spans, K1, frozenset({"PERSON"}))
    denied = redact_question(q, spans, K1, frozenset())
    doc_token = token_for(K1, "PERSON", "John Smith")[0]
    assert find_tokens(allowed.text) == [doc_token]       # matches chunks
    assert find_tokens(denied.text) != [doc_token]        # matches nothing
    assert "John" not in denied.text
    assert set(denied.own.values()) == {"John Smith"}     # still readable for the asker


_word = st.text(alphabet="abcdefghij ", min_size=1, max_size=12)


@settings(max_examples=40, deadline=None)
@given(parts=st.lists(st.tuples(_word, st.booleans()), min_size=1, max_size=400))
def test_chunker_never_splits_tokens(parts: list[tuple[str, bool]]) -> None:
    tok = bracket(token_for(K1, "PERSON", "x")[0])
    text = " ".join(f"{w} {tok}." if t else f"{w}." for w, t in parts)
    chunks = Chunker(chunk_size=64, chunk_overlap=8).split(text)
    for c in chunks:
        # Every "[PERSON_" start in a chunk is a whole token.
        assert c.count("[PERSON_") == len(find_tokens(c))
    if any(t for _, t in parts):
        assert any(find_tokens(c) for c in chunks)


def test_propagate_hides_every_copy_and_surname() -> None:
    texts = ["Contact Priyanka Raghunathan today.", "Priyanka Raghunathan joined.",
             "Raghunathan said hi. Raghunathanx is a different word."]
    spans: list[list[tuple[int, int, str, float]]] = [[], [(0, 20, "PERSON", 0.85)], []]
    out = propagate(texts, spans)
    assert [(s[0], s[1]) for s in out[0]] == [(8, 28)]
    assert [(s[0], s[1]) for s in out[1]] == [(0, 20)]
    assert [(s[0], s[1]) for s in out[2]] == [(0, 11)]  # word boundary: not "Raghunathanx"
    assert all(s[2] == "PERSON" for sp in out for s in sp)


def test_propagate_ignores_short_values() -> None:
    texts = ["Al met Bo.", "Al is here."]
    out = propagate(texts, [[(0, 2, "PERSON", 0.9)], []])
    assert out[1] == []


def test_name_candidates_extend_partial_spans_but_not_cut_them() -> None:
    text = "Phone: 785-277-4932 Joel Jackson joined Acme Corp today."
    partial = [(25, 32, "PERSON", 0.85)]  # spaCy found only "Jackson"
    cands = name_candidates([text], [partial], K1)
    found = {text[a:b] for places in cands.values() for _, a, b in places}
    assert "Joel Jackson" in found            # contains the partial span: kept
    assert token_for(K1, "PERSON", "Joel Jackson")[0] in cands
    cut = name_candidates([text], [[(25, 39, "PERSON", 0.85)]], K1)  # "Jackson joined"
    assert not any(text[a:b] == "Joel Jackson" for p in cut.values() for _, a, b in p)
    assert "Acme Corp" in {text[a:b] for p in cands.values() for _, a, b in p}
