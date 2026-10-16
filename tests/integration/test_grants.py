"""Each DB login can do only its own job. Audit is append-only, even for the owner."""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from govrag.db import Pools, scoped
from tests.conftest import insert_doc


async def test_app_cannot_touch_vault(pools: Pools, owner: asyncpg.Connection) -> None:
    for sql in (
        "SELECT * FROM vault.pii_tokens",
        "SELECT * FROM vault.tenant_enc_keys",
        "SELECT * FROM vault.token_refs",
    ):
        async with scoped(pools.app, "acme", ["hr"], acquire_timeout_s=1) as c:
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                await c.fetch(sql)


async def test_app_cannot_write_chunks(pools: Pools, owner: asyncpg.Connection) -> None:
    async with scoped(pools.app, "acme", ["hr"], acquire_timeout_s=1) as c:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await c.execute("DELETE FROM core.chunks")


async def test_ingest_cannot_read_ciphertext(pools: Pools) -> None:
    async with scoped(pools.ingest, "acme", acquire_timeout_s=1) as c:
        await c.fetch("SELECT token, value_mac FROM vault.pii_tokens")  # allowed columns
    async with scoped(pools.ingest, "acme", acquire_timeout_s=1) as c:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await c.fetch("SELECT ciphertext FROM vault.pii_tokens")


async def test_vault_login_reads_only_vault(pools: Pools) -> None:
    async with scoped(pools.vault, "acme", acquire_timeout_s=1) as c:
        await c.fetch("SELECT token, ciphertext FROM vault.pii_tokens")
    async with scoped(pools.vault, "acme", acquire_timeout_s=1) as c:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await c.fetch("SELECT * FROM core.chunks")


def _audit_args(tenant: str) -> tuple[str, uuid.UUID, str, str, str]:
    return (tenant, uuid.uuid4(), "u1", "retrieve", "ok")


AUDIT_INSERT = (
    "INSERT INTO audit.events (tenant_id, request_id, user_id, action, outcome)"
    " VALUES ($1, $2, $3, $4, $5)"
)


async def test_audit_insert_only_own_tenant(pools: Pools) -> None:
    async with scoped(pools.app, "acme", ["hr"], acquire_timeout_s=1) as c:
        await c.execute(AUDIT_INSERT, *_audit_args("acme"))
    async with scoped(pools.app, "acme", ["hr"], acquire_timeout_s=1) as c:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):  # RLS WITH CHECK
            await c.execute(AUDIT_INSERT, *_audit_args("globex"))


async def test_audit_is_append_only(pools: Pools, owner: asyncpg.Connection) -> None:
    async with scoped(pools.app, "acme", ["hr"], acquire_timeout_s=1) as c:
        for sql in ("UPDATE audit.events SET outcome = 'ok'", "DELETE FROM audit.events"):
            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with c.transaction():
                    await c.execute(sql)
    # The owner is a superuser, so grants don't stop it. The trigger does.
    for sql in (
        "UPDATE audit.events SET outcome = 'error'",
        "DELETE FROM audit.events",
        "TRUNCATE audit.events",
    ):
        with pytest.raises(asyncpg.InsufficientPrivilegeError, match="append-only"):
            await owner.execute(sql)


async def test_chunk_roles_must_be_subset_of_document(owner: asyncpg.Connection) -> None:
    with pytest.raises(asyncpg.CheckViolationError, match="subset"):
        await insert_doc(owner, "acme", ["intern"], [("wider than doc", ["intern", "hr"])])


async def test_empty_roles_rejected(owner: asyncpg.Connection) -> None:
    with pytest.raises(asyncpg.CheckViolationError):
        await insert_doc(owner, "acme", [], [])
