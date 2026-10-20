"""Token revocation: reject a user's tokens issued before a cut-off.

All active revocations live in memory (a dict lookup per request). Rows older than the max
token lifetime are skipped on load: any token they could affect has expired anyway, so the
set stays small no matter how long the service runs.

Freshness is the same as the policy cache: NOTIFY reloads in milliseconds, a 30 s TTL
covers lost notifications, and if the list is stale *and* the DB is down, requests fail
(503) rather than skip the check.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import asyncpg

from govrag import audit
from govrag.auth import Principal
from govrag.db import Pools, scoped
from govrag.errors import BadRequest, Forbidden, Unauthorized, Unavailable
from govrag.notify import Listener
from govrag.policy.store import ADMIN_ROLE

CHANNEL = "revocations_changed"


class Revocations:
    def __init__(
        self, pools: Pools, listener: Listener, *, max_token_age_s: int, ttl_s: float = 30.0
    ) -> None:
        self._pools = pools
        self._max_age = max_token_age_s
        self._ttl = ttl_s
        self._cut: dict[tuple[str, str], float] = {}
        self._loaded_at = float("-inf")
        self._lock = asyncio.Lock()
        listener.subscribe(CHANNEL, self._on_notify, on_connect=self._mark_stale)

    def _on_notify(self, _c: object, _pid: int, _channel: str, _payload: str) -> None:
        self._mark_stale()

    def _mark_stale(self) -> None:
        self._loaded_at = float("-inf")

    async def _load(self) -> None:
        async with self._lock:  # single-flight: one reload however many requests miss
            if time.monotonic() - self._loaded_at < self._ttl:
                return
            try:
                async with self._pools.app.acquire(timeout=self._pools.acquire_timeout_s) as c:
                    rows = await c.fetch(
                        "SELECT tenant_id, subject, extract(epoch FROM not_before) AS nb"
                        " FROM core.revocations"
                        " WHERE not_before > now() - make_interval(secs => $1)",
                        float(self._max_age),
                    )
            except (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError) as e:
                raise Unavailable("revocations_unavailable") from e
            self._cut = {(r["tenant_id"], r["subject"]): float(r["nb"]) for r in rows}
            self._loaded_at = time.monotonic()

    async def check(self, p: Principal) -> None:
        if time.monotonic() - self._loaded_at >= self._ttl:
            await self._load()
        cut = self._cut.get((p.tenant_id, p.user_id))
        if cut is not None and p.issued_at < cut:
            raise Unauthorized("token_revoked")

    async def revoke(self, admin: Principal, user_id: str) -> float:
        if not admin.has(ADMIN_ROLE):
            raise Forbidden("policy_admin_required")
        try:
            async with scoped(self._pools.app, admin.tenant_id, admin.roles,
                              acquire_timeout_s=self._pools.acquire_timeout_s) as c:
                nb = await c.fetchval(
                    "INSERT INTO core.revocations (tenant_id, subject, not_before, created_by)"
                    " VALUES ($1, $2, now(), $3) ON CONFLICT (tenant_id, subject) DO UPDATE"
                    " SET not_before = now(), created_by = EXCLUDED.created_by"
                    " RETURNING extract(epoch FROM not_before)",
                    admin.tenant_id, user_id, admin.user_id,
                )
                await audit.record(c, audit.AuditEvent(
                    tenant_id=admin.tenant_id, request_id=uuid.uuid4(), user_id=admin.user_id,
                    action="revoke",
                ))
        except asyncpg.CheckViolationError as e:
            raise BadRequest("bad_user_id") from e
        self._cut[(admin.tenant_id, user_id)] = float(nb)  # this instance: immediately
        return float(nb)
