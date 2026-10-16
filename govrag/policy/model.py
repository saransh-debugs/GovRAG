"""The policy document. Validated on every write; a bad policy is never stored."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from govrag.db import IDENT
from govrag.pii.entities import KNOWN, OPTIONAL_REDACT

# Models the server knows how to call. A tenant picks from this list; it can't name an
# arbitrary endpoint, so a policy edit can't route a tenant's data to a new vendor.
SERVER_MODELS: frozenset[str] = frozenset({"claude-haiku-5.5", "gemini-2.5-flash-lite", "fake"})


def _role(v: str) -> str:
    if not IDENT.match(v):
        raise ValueError("bad role id")
    return v


def _known_type(v: str) -> str:
    if v not in KNOWN:
        raise ValueError(f"unknown PII type {v!r}")
    return v


Role = Annotated[str, AfterValidator(_role)]
PiiType = Annotated[str, AfterValidator(_known_type)]


class LlmPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # First = main, the rest = backups, tried in order.
    models: tuple[str, ...] = ("claude-haiku-5.5", "gemini-2.5-flash-lite")
    region: Literal["us", "eu"] = "us"

    @field_validator("models")
    @classmethod
    def _models(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not 1 <= len(v) <= 3 or len(set(v)) != len(v):
            raise ValueError("1-3 distinct models")
        bad = [m for m in v if m not in SERVER_MODELS]
        if bad:
            raise ValueError(f"models not on the server allowlist: {bad}")
        return v


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # role -> PII types that role may see as real values. Missing role = sees none.
    unmask: dict[Role, frozenset[PiiType]] = Field(default_factory=dict, max_length=256)
    extra_redact: frozenset[PiiType] = frozenset()
    llm: LlmPolicy = LlmPolicy()

    @field_validator("extra_redact")
    @classmethod
    def _optional_only(cls, v: frozenset[str]) -> frozenset[str]:
        # The default list is always redacted; listing it here would suggest it's optional.
        bad = v - OPTIONAL_REDACT
        if bad:
            raise ValueError(f"already redacted by default: {sorted(bad)}")
        return v

    def may_unmask(self, roles: frozenset[str]) -> frozenset[str]:
        out: set[str] = set()
        for r in roles:
            out |= self.unmask.get(r, frozenset())
        return frozenset(out)


# A tenant with no policy row: nobody unmasks anything. The safe default.
DEFAULT_POLICY = Policy()


class ActivePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int  # 0 = DEFAULT_POLICY, nothing stored yet
    policy: Policy
