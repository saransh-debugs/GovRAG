"""PII types we know about, and the ones always redacted at ingest."""

from __future__ import annotations

# Always hidden at ingest, for every tenant. Policies only decide who may *unmask* these.
# That's what makes policy edits instant: loosening or tightening unmask rules never
# requires reprocessing stored chunks.
DEFAULT_REDACT: frozenset[str] = frozenset(
    {
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IBAN_CODE",
        "IP_ADDRESS",
        "US_BANK_NUMBER",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "US_ITIN",
        "MEDICAL_LICENSE",
    }
)

# Noisier types a tenant can opt into with `extra_redact`. Adding one triggers a
# background re-scan of the tenant's existing chunks (govrag/policy/rescan.py).
OPTIONAL_REDACT: frozenset[str] = frozenset(
    {"LOCATION", "DATE_TIME", "NRP", "ORGANIZATION", "URL", "CRYPTO", "MAC_ADDRESS"}
)

KNOWN: frozenset[str] = DEFAULT_REDACT | OPTIONAL_REDACT

# Per-type minimum confidence. Presidio scores phone numbers ~0.4 without context words,
# and names from spaCy at ~0.85. One global threshold either misses phones or floods
# false-positive names.
MIN_SCORE: dict[str, float] = {
    "PHONE_NUMBER": 0.4,
    "PERSON": 0.6,
    "LOCATION": 0.6,
    "ORGANIZATION": 0.7,
    "DATE_TIME": 0.6,
    "URL": 0.6,
}
DEFAULT_MIN_SCORE = 0.5
