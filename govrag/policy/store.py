"""Policy versions in Postgres: read the live one, write a new one, roll back.

Writes use optimistic concurrency: the caller says which version it's replacing
(`expected_version`). If another admin saved first, the write fails with Conflict instead
of silently overwriting their change.
"""

from __future__ import annotations

import uuid

import asyncpg

from govrag import audit
from govrag.auth import Principal
from govrag.db import Pools, scoped
from govrag.errors import Conflict, Forbidden, NotFound
from govrag.policy.model import DEFAULT_POLICY, ActivePolicy, Policy

ADMIN_ROLE = "policy_admin"


async def read_active(conn: asyncpg.Connection, tenant_id: str) -> ActivePolicy:
    row = await conn.fetchrow(
        "SELECT a.version, v.doc FROM policy.active a"
        " JOIN policy.versions v ON v.tenant_id = a.tenant_id AND v.version = a.version"
        " WHERE a.tenant_id = $1",
        tenant_id,
    )
    if row is None:
        return ActivePolicy(version=0, policy=DEFAULT_POLICY)
    return ActivePolicy(version=row["version"], policy=Policy.model_validate_json(row["doc"]))


class PolicyStore:
    def __init__(self, pools: Pools) -> None:
        self._pools = pools

    async def get(self, p: Principal) -> ActivePolicy:
        async with scoped(self._pools.app, p.tenant_id, p.roles,
                          acquire_timeout_s=self._pools.acquire_timeout_s) as c:
            return await read_active(c, p.tenant_id)

    async def versions(self, p: Principal) -> list[tuple[int, str]]:
        if not p.has(ADMIN_ROLE):
            raise Forbidden("policy_admin_required")
        async with scoped(self._pools.app, p.tenant_id, p.roles,
                          acquire_timeout_s=self._pools.acquire_timeout_s) as c:
            rows = await c.fetch(
                "SELECT version, created_by FROM policy.versions WHERE tenant_id = $1"
                " ORDER BY version DESC LIMIT 100",
                p.tenant_id,
            )
        return [(r["version"], r["created_by"]) for r in rows]

    async def put(
        self,
        p: Principal,
        policy: Policy,
        *,
        expected_version: int,
        request_id: uuid.UUID | None = None,
    ) -> ActivePolicy:
        """Save `policy` as a new version and make it live. Returns the new ActivePolicy.

        If extra_redact gained types, a re-scan job for the tenant's old chunks is queued
        in the same transaction.
        """
        if not p.has(ADMIN_ROLE):
            raise Forbidden("policy_admin_required")
        async with scoped(self._pools.app, p.tenant_id, p.roles,
                          acquire_timeout_s=self._pools.acquire_timeout_s) as c:
            # Lock the tenant's pointer row so two concurrent saves serialize here.
            cur = await c.fetchrow(
                "SELECT version FROM policy.active WHERE tenant_id = $1 FOR UPDATE", p.tenant_id
            )
            current = cur["version"] if cur else 0
            if current != expected_version:
                raise Conflict("policy_version_conflict")
            old = await read_active(c, p.tenant_id)
            new_version = (
                await c.fetchval(
                    "SELECT coalesce(max(version), 0) FROM policy.versions WHERE tenant_id = $1",
                    p.tenant_id,
                )
                + 1
            )
            try:
                await c.execute(
                    "INSERT INTO policy.versions (tenant_id, version, doc, created_by)"
                    " VALUES ($1, $2, $3::jsonb, $4)",
                    p.tenant_id, new_version, policy.model_dump_json(), p.user_id,
                )
            except asyncpg.UniqueViolationError:
                # The tenant's first-ever save has no `active` row to lock, so two
                # concurrent first saves both get here. The loser gets a Conflict.
                raise Conflict("policy_version_conflict") from None
            await self._point(c, p, new_version)
            added = policy.extra_redact - old.policy.extra_redact
            if added:
                await c.execute(
                    "INSERT INTO core.rescan_jobs (tenant_id, types, target_rev)"
                    " VALUES ($1, $2, $3)",
                    p.tenant_id, sorted(added), new_version,
                )
            await audit.record(c, audit.AuditEvent(
                tenant_id=p.tenant_id, request_id=request_id or uuid.uuid4(), user_id=p.user_id,
                action="policy_change", policy_version=new_version, prev_policy_version=current,
            ))
        return ActivePolicy(version=new_version, policy=policy)

    async def rollback(
        self,
        p: Principal,
        to_version: int,
        *,
        expected_version: int,
        request_id: uuid.UUID | None = None,
    ) -> ActivePolicy:
        if not p.has(ADMIN_ROLE):
            raise Forbidden("policy_admin_required")
        async with scoped(self._pools.app, p.tenant_id, p.roles,
                          acquire_timeout_s=self._pools.acquire_timeout_s) as c:
            cur = await c.fetchrow(
                "SELECT version FROM policy.active WHERE tenant_id = $1 FOR UPDATE", p.tenant_id
            )
            current = cur["version"] if cur else 0
            if current != expected_version:
                raise Conflict("policy_version_conflict")
            doc = await c.fetchval(
                "SELECT doc FROM policy.versions WHERE tenant_id = $1 AND version = $2",
                p.tenant_id, to_version,
            )
            if doc is None:
                raise NotFound("policy_version_not_found")
            await self._point(c, p, to_version)
            await audit.record(c, audit.AuditEvent(
                tenant_id=p.tenant_id, request_id=request_id or uuid.uuid4(), user_id=p.user_id,
                action="policy_rollback", policy_version=to_version, prev_policy_version=current,
            ))
        return ActivePolicy(version=to_version, policy=Policy.model_validate_json(doc))

    @staticmethod
    async def _point(c: asyncpg.Connection, p: Principal, version: int) -> None:
        await c.execute(
            "INSERT INTO policy.active (tenant_id, version, updated_by) VALUES ($1, $2, $3)"
            " ON CONFLICT (tenant_id) DO UPDATE"
            " SET version = EXCLUDED.version, updated_by = EXCLUDED.updated_by,"
            "     updated_at = now()",
            p.tenant_id, version, p.user_id,
        )
