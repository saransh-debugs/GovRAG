"""PII tokens: `[PERSON_3f9a1c2e4b7d9a01]`.

token = TYPE + "_" + first 16 hex chars (64 bits) of HMAC-SHA256(tenant_key, TYPE \\0 value).

Why HMAC and not `[PERSON_1]` numbering:
  - Numbering restarts per document, so `[PERSON_1]` means two different people in two
    retrieved chunks. Rehydration would put back the wrong name.
  - Same value -> same token (within a tenant), so the LLM can tell two chunks are about
    the same person, and a redacted question still matches redacted chunks in search.
  - Keyed, so nobody without the tenant key can compute "the token for John Smith".
Why 64 bits: tokens are copied by an LLM, so shorter is better. 64 bits gives ~1e-8
collision odds at 1M distinct values per type; the vault stores the full MAC and
ingest checks it, so a collision fails loudly instead of mixing up two people.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata

TOKEN_HEX = 16
# Bracketed form, as it appears in text. Type is uppercase letters/digits/underscores.
TOKEN_RE = re.compile(r"\[([A-Z][A-Z0-9_]{0,40})_([0-9a-f]{16})\]")
# Bare form (no brackets) - only used to neutralize user input.
_BARE_RE = re.compile(r"\b([A-Z][A-Z0-9_]{0,40})_([0-9a-f]{16})\b")

# For structured types, formatting doesn't change identity: 123-45-6789 == 123456789.
_DIGITS_ONLY = {"US_SSN", "CREDIT_CARD", "PHONE_NUMBER", "US_BANK_NUMBER", "US_ITIN"}


def normalize(entity_type: str, value: str) -> str:
    v = unicodedata.normalize("NFKC", value).strip()
    if entity_type in _DIGITS_ONLY:
        return "".join(ch for ch in v if ch.isdigit())
    if entity_type == "EMAIL_ADDRESS":
        return v.lower()
    # Names, places: case and runs of whitespace don't matter. "john  SMITH" == "John Smith".
    return " ".join(v.casefold().split())


def value_mac(hmac_key: bytes, entity_type: str, value: str) -> bytes:
    msg = entity_type.encode() + b"\x00" + normalize(entity_type, value).encode()
    return hmac.new(hmac_key, msg, hashlib.sha256).digest()


def token_for(hmac_key: bytes, entity_type: str, value: str) -> tuple[str, bytes]:
    """Returns (token_id without brackets, full 32-byte MAC)."""
    mac = value_mac(hmac_key, entity_type, value)
    return f"{entity_type}_{mac.hex()[:TOKEN_HEX]}", mac


def random_token(entity_type: str) -> str:
    """A token that matches nothing. Used for question PII the asker isn't allowed to see,
    so they can't confirm a guessed name by checking whether search matches it."""
    return f"{entity_type}_{os.urandom(TOKEN_HEX // 2).hex()}"


def bracket(token: str) -> str:
    return f"[{token}]"


def find_tokens(text: str) -> list[str]:
    """Token IDs (no brackets) in order of appearance, duplicates kept."""
    return [f"{m.group(1)}_{m.group(2)}" for m in TOKEN_RE.finditer(text)]


def entity_type_of(token: str) -> str:
    return token.rsplit("_", 1)[0]


def neutralize(text: str) -> str:
    """Break token-shaped strings in user input so they can never be rehydrated.

    "[PERSON_3f9a1c2e4b7d9a01]" -> "(PERSON-3f9a1c2e4b7d9a01)". The rehydrator also only
    accepts tokens from this request's retrieved chunks; this is the second check.
    """
    text = TOKEN_RE.sub(lambda m: f"({m.group(1)}-{m.group(2)})", text)
    return _BARE_RE.sub(lambda m: f"{m.group(1)}-{m.group(2)}", text)
