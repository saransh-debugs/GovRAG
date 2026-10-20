"""PII in labeled fields: `last_name: "Zölly"`, `"PassportNumber": "791797407"`,
`<strong>Passport:</strong> 384542560`, `**Driver's License**: F42397376`.

Why: on a public third-party dataset (eval/pii_public.py), Presidio + spaCy found only
~40% of names and ~10-30% of passports, licenses and phones. Most misses sat behind a
label. NER reads prose; forms, JSON, YAML, HTML and markdown tables are where business
documents keep their PII. The label is strong evidence, so it decides the type.

Pure function, no Presidio: runs inside the worker next to it, and is unit-testable.
"""

from __future__ import annotations

import re

Span = tuple[int, int, str, float]
SCORE = 0.95  # above NER and pattern scores: a label beats a guess

# Label keyword -> type. Checked in order; the first match wins ("user name" before "name").
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.I), t) for p, t in [
        (r"pass(?:word|code|phrase)?$|^pwd$|secret|api[ _-]?key|access[ _-]?token|^pin$",
         "CREDENTIAL"),
        (r"user[ _-]?(?:name|id)|login|handle|screen[ _-]?name", "USERNAME"),
        (r"passport", "PASSPORT"),
        (r"driv(?:er|ing)'?s?[ _-]?licen[cs]e|^dl$|licen[cs]e[ _-]?(?:no|number)",
         "DRIVER_LICENSE"),
        (r"^ssn|ssn$|social[ _-]?security", "US_SSN"),
        (r"social[ _-]?(?:number|insurance)|national[ _-]?(?:id|insurance)"
         r"|id[ _-]?(?:card|number|no)|identity[ _-]?(?:card|number)|tax[ _-]?id"
         r"|(?:student|employee|member|patient|customer|staff|national)[ _-]?(?:id|number|no)$",
         "NATIONAL_ID"),
        (r"e-?mail", "EMAIL_ADDRESS"),
        (r"phone|^tel|mobile|cell|fax|contact[ _-]?(?:number|no)", "PHONE_NUMBER"),
        (r"birth|^dob$|^bod$", "DATE_OF_BIRTH"),
        (r"street|address|^addr", "STREET_ADDRESS"),
        (r"(?:first|given|last|sur|middle|family|full|maiden|middle)[ _-]?name"
         r"|^name$|^surname$|^forename$|^firstname$|^lastname$|^givenname$",
         "PERSON"),
        (r"^ip(?:[ _-]?address)?$", "IP_ADDRESS"),
        (r"card[ _-]?(?:number|no)|credit[ _-]?card|^iban$|account[ _-]?(?:number|no)",
         "CREDIT_CARD"),
    ]
]

# `label` then a separator (":" or "=", or a closing tag), then the value.
#   last_name: "Zölly"        "PassportNumber": "791797407"      Telephone: 043-5237.5019
#   <strong>Passport:</strong> 384542560                          **Driver's License**: F42
_LABELED = re.compile(
    r"""(?P<label>[A-Za-z][A-Za-z0-9 _'.\-]{0,40}?)          # label words
        ["'*]{0,2}[ \t]*(?:</[A-Za-z][^>]{0,20}>)?[ \t]*       # closing quote / ** / </b>
        [:=][ \t]*(?:</[A-Za-z][^>]{0,20}>[ \t]*)?             # separator (+ </strong>)
        (?:<[A-Za-z][^>]{0,20}>[ \t]*)*(?P<open>["'*\[]{0,2})[ \t]*  # opening <b>/quote/**/[
        (?P<value>[^"'\n<>,*|{}\[\]]{1,80}?)                     # the value
        (?=[ \t]{2,}|[ \t]*["'*<\n,|}\]]|[ \t]+-[ \t]|[ \t]*$)       # ends: 2+ spaces, quote,
                                                                # tag, newline, " - ", end
    """, re.X | re.M)
# Strong labels that often come without a colon: "Passport Number 345776329".
_BARE_LABEL = re.compile(
    r"(?i:(?P<label>passport(?:[ _-]?(?:number|no))?|social(?:[ _-]security)?[ _-]number"
    r"|driver'?s?[ _-]licen[cs]e(?:[ _-]number)?|id[ _-]card(?:[ _-]number)?|phone[ _-]number))"
    # The value is case-sensitive on purpose: IDs are digits and capitals, so the match
    # stops at the next ordinary word ("036 617 3108 and Passport ..." -> "036 617 3108").
    r"[ \t]+(?:is[ \t]+)?(?P<value>[A-Z0-9][A-Z0-9.\-]*(?:[ .\-][A-Z0-9]+)*)\b")
# <last_name>Zölly</last_name>
_XML = re.compile(r"<(?P<label>[A-Za-z][\w\-.]{0,40})>(?P<value>[^<\n]{1,80})</(?P=label)>")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def _type_of(label: str) -> str | None:
    words = _CAMEL.sub(" ", label).replace("_", " ").replace("-", " ").strip(" .'\"*")
    words = re.sub(r"\s+", " ", words)
    # Try the whole label, then its last two words, then its last word: "Witness Driver's
    # License" -> "driver's license", "billing_address" -> "address".
    parts = words.split(" ")
    for cand in (words, " ".join(parts[-2:]), parts[-1]):
        for pat, t in _RULES:
            if pat.search(cand):
                return t
    return None


_QUOTED = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_YEAR_TAIL = re.compile(r",?[ \t]+\d{2,4}\b")
# Identifier-shaped: tokens of digits/capitals; after the first, each token needs a digit.
# "LISEB 860120 9 744" and "0042-29 276.2523" match whole; "374-11-3613. Contact" stops.
_ID = re.compile(r"[+(]?[A-Z0-9][A-Z0-9.\-()/]*(?:[ ][A-Z0-9.\-()/]*\d[A-Z0-9.\-()/]*)*")
_NAME = re.compile(r"[A-ZÀ-ÖØ-Þ][\w'\-]*(?:[ ](?:[A-ZÀ-ÖØ-Þ][\w'\-]*|de|da|van|von|der|del"
                   r"|la|le|di|du|bin|al)){0,4}")
_SENTENCE_END = re.compile(r"[.!?][ \t]+(?=[A-Z])")
_ID_TYPES = {"US_SSN", "NATIONAL_ID", "PASSPORT", "DRIVER_LICENSE", "PHONE_NUMBER",
             "CREDIT_CARD", "IP_ADDRESS"}


def _shape(text: str, t: str, a: int, b: int) -> int:
    """Cut the generic value down to what this type looks like, so a value never runs on
    into the next sentence and swallows a name ("SSN on file: 374-11-3613. Contact Juan")."""
    if t in _ID_TYPES:
        m = _ID.match(text, a, b)
        return m.end() if m else b
    if t == "PERSON":
        m = _NAME.match(text, a, b)
        return m.end() if m else b
    m = _SENTENCE_END.search(text, a, b)
    return m.start() if m else b


def _extend(text: str, t: str, a: int, b: int, opener: str) -> int:
    """Type-specific value end. Secrets contain quotes and commas; dates contain commas."""
    if t in ("CREDENTIAL", "USERNAME"):
        if opener.startswith('"'):
            m = _QUOTED.match(text, a - 1)  # JSON string, \" escapes included
            if m:
                return m.end(1)
        end = a
        while end < len(text) and not text[end].isspace():
            end += 1
        # A table or markup closer glued to the value isn't part of the secret.
        word = text[a:end]
        for closer in ("</td>", "**", "|"):
            if word.endswith(closer) and len(word) > len(closer):
                word = word[: -len(closer)]
        return max(a + len(word), b)
    if t == "DATE_OF_BIRTH":
        m = _YEAR_TAIL.match(text, b)
        if m:
            return m.end()
    return b


def field_spans(text: str, entities: frozenset[str] | set[str] | list[str]) -> list[Span]:
    wanted = set(entities)
    out: list[Span] = []
    for rx in (_LABELED, _XML, _BARE_LABEL):
        for m in rx.finditer(text):
            t = _type_of(m.group("label"))
            if t is None or t not in wanted:
                continue
            a, b = m.span("value")
            opener = m.groupdict().get("open") or ""
            if t not in ("CREDENTIAL", "USERNAME"):
                b = _shape(text, t, a, b)
            b = _extend(text, t, a, b, opener)
            value = text[a:b]
            # A bare label with no real value ("Password: ", "Name: N/A") isn't PII.
            if not value.strip() or value.strip().lower() in {"n/a", "na", "none", "null", "-"}:
                continue
            # Trim trailing punctuation so "Zölly." hides "Zölly".
            while b > a and text[b - 1] in " .;":
                b -= 1
            out.append((a, b, t, SCORE))
    return out
