"""Durable audit events (Postgres, insert-only).

`record()` runs inside the caller's transaction, so the audit row commits or rolls back
with the work it describes. If the insert fails, the caller's request fails. There is no
"log it later" path: an access we can't record is an access we don't allow.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import asyncpg
from opentelemetry import trace

Action = Literal[
    "retrieve", "rehydrate", "ingest", "delete", "denied",
    "policy_change", "policy_rollback", "rescan", "shred", "dry_run",
]
Outcome = Literal["ok", "denied", "error"]


@dataclass(frozen=True, slots=True)
class AuditEvent:
    tenant_id: str
    request_id: uuid.UUID
    user_id: str
    action: Action
    outcome: Outcome = "ok"
    document_ids: tuple[uuid.UUID, ...] = ()
    chunk_ids: tuple[uuid.UUID, ...] = ()
    pii_types_present: frozenset[str] = field(default_factory=frozenset)
    pii_types_unmasked: frozenset[str] = field(default_factory=frozenset)
    policy_version: int | None = None
    prev_policy_version: int | None = None
    model: str | None = None
    error_code: str | None = None


def _ids(xs: Iterable[uuid.UUID]) -> list[uuid.UUID]:
    return list(dict.fromkeys(xs))  # dedupe, keep order


async def record(conn: asyncpg.Connection, ev: AuditEvent) -> None:
    # Mirror the event onto the current span for debugging. Same fields, no text.
    span = trace.get_current_span()
    span.set_attribute("govrag.audit.action", ev.action)
    await conn.execute(
        """
        INSERT INTO audit.events (tenant_id, request_id, user_id, action, outcome,
            document_ids, chunk_ids, pii_types_present, pii_types_unmasked,
            policy_version, prev_policy_version, model, error_code)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
        ev.tenant_id,
        ev.request_id,
        ev.user_id,
        ev.action,
        ev.outcome,
        _ids(ev.document_ids),
        _ids(ev.chunk_ids),
        sorted(ev.pii_types_present),
        sorted(ev.pii_types_unmasked),
        ev.policy_version,
        ev.prev_policy_version,
        ev.model,
        ev.error_code,
    )
