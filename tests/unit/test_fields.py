"""Labeled-field PII finder across the formats business documents actually use."""

from __future__ import annotations

import pytest

from govrag.pii.entities import DEFAULT_REDACT
from govrag.pii.fields import field_spans


def found(text: str) -> dict[str, str]:
    return {text[a:b]: t for a, b, t, _ in field_spans(text, DEFAULT_REDACT)}


@pytest.mark.parametrize(("text", "value", "etype"), [
    ('testing_time: "23"\n  last_name: "Zölly"\n', "Zölly", "PERSON"),
    ('{"PassportNumber": "791797407", "Country": "GB"}', "791797407", "PASSPORT"),
    ("<p><strong>Passport:</strong> 384542560</p>", "384542560", "PASSPORT"),
    ("<p><strong>Telephone:</strong> 043-5237.5019</p>", "043-5237.5019", "PHONE_NUMBER"),
    ("- **Witness Driver's License**: LISEB 860120 9 744\n", "LISEB 860120 9 744",
     "DRIVER_LICENSE"),
    ("Driver's License: F42397376  2. Next", "F42397376", "DRIVER_LICENSE"),
    ('"IDCARD": "9227208827",', "9227208827", "NATIONAL_ID"),
    ("Social Security Number: **461-33-9220**", "461-33-9220", "US_SSN"),
    ("<last_name>Zölly</last_name>", "Zölly", "PERSON"),
    ("username = jdoe_88\n", "jdoe_88", "USERNAME"),
    ('password: "5n5JWo%1"', "5n5JWo%1", "CREDENTIAL"),
    ("Date of Birth: 17th February 1946\n", "17th February 1946", "DATE_OF_BIRTH"),
    ("billing_address: 42 Elm Street\n", "42 Elm Street", "STREET_ADDRESS"),
    ("First Name: Karin\nLast Name: Rossi\n", "Karin", "PERSON"),
    ("Email: a.b@c.io\n", "a.b@c.io", "EMAIL_ADDRESS"),
    # Round two, from the public dataset's remaining misses:
    ("Time: [1:02am], Name: [Ruqaya], Name: [Yoliyana]", "Ruqaya", "PERSON"),
    ('- **Password:** 2#r"Agb\n', '2#r"Agb', "CREDENTIAL"),
    ('"password": "0\\">vh/Flp@C",\n', '0\\">vh/Flp@C', "CREDENTIAL"),
    ("- Date of Birth: September 13th, 1987\n", "September 13th, 1987", "DATE_OF_BIRTH"),
    ("- Student ID: 90175953503326 | - Last Name: Sader", "90175953503326", "NATIONAL_ID"),
    ("   Contact Number: 0042-29 276.2523\n", "0042-29 276.2523", "PHONE_NUMBER"),
    ("Social Number 036 617 3108 and Passport Number 345776329 | x", "345776329", "PASSPORT"),
    ("Social Number 036 617 3108 and more", "036 617 3108", "NATIONAL_ID"),
    # Values stop where the type's shape ends, not at the next sentence's name.
    ("SSN on file: 374-11-3613. Contact Juan Wood today.", "374-11-3613", "US_SSN"),
    ("Phone: 276-018-9555 Contact Eric Simpson today.", "276-018-9555", "PHONE_NUMBER"),
    ("Name: Karin Rossi joined the team.", "Karin Rossi", "PERSON"),
    # YAML: a key with no value must not swallow the next line.
    ('Psychotherapist_C:\n  Phone_Number: "+34.28168-5293"\n', "+34.28168-5293",
     "PHONE_NUMBER"),
])
def test_labeled_values(text: str, value: str, etype: str) -> None:
    assert found(text).get(value) == etype, found(text)


def test_unlabeled_and_empty_values_are_ignored() -> None:
    assert found("Status: approved\nCountry: GB\nPassword: N/A\n") == {}
    assert found("The meeting is at 10:30 today.") == {}


def test_types_not_requested_are_skipped() -> None:
    assert field_spans("Password: hunter2", {"PERSON"}) == []
