"""Row-Level Security: tenant + role isolation enforced by Postgres itself."""

from __future__ import annotations

import asyncpg
import pytest

from govrag.db import Pools, roles_literal, scoped
from tests.conftest import insert_doc


@pytest.fixture(scope="module")
async def seeded(owner: asyncpg.Connection) -> dict[str, object]:
    _, acme = await insert_doc(
        owner, "acme", ["hr", "intern"],
        [("acme hr only", ["hr"]), ("acme everyone", ["hr", "intern"])],
    )
    _, globex = await insert_doc(owner, "globex", ["hr"], [("globex hr", ["hr"])])
    return {"acme": acme, "globex": globex}


async def _count(pools: Pools, tenant: str, roles: list[str], where: str = "true") -> int:
    async with scoped(pools.app, tenant, roles, acquire_timeout_s=1) as c:
        n: int = await c.fetchval(f"SELECT count(*) FROM core.chunks WHERE {where}")  # noqa: S608
        return n


async def test_nothing_visible_without_context(pools: Pools, seeded: object) -> None:
    async with pools.app.acquire() as c:
        assert await c.fetchval("SELECT count(*) FROM core.chunks") == 0
        assert await c.fetchval("SELECT count(*) FROM core.documents") == 0


async def test_tenant_without_roles_sees_nothing(pools: Pools, seeded: object) -> None:
    assert await _count(pools, "acme", []) == 0


async def test_role_filtering(pools: Pools, seeded: object) -> None:
    assert await _count(pools, "acme", ["intern"]) == 1
    assert await _count(pools, "acme", ["hr"]) == 2
    assert await _count(pools, "acme", ["intern", "hr"]) == 2
    assert await _count(pools, "acme", ["finance"]) == 0


async def test_cross_tenant_invisible_even_when_asked_for(pools: Pools, seeded: object) -> None:
    # Same role name in the other tenant, and an explicit WHERE for the other tenant.
    assert await _count(pools, "acme", ["hr"], "tenant_id = 'globex'") == 0
    assert await _count(pools, "globex", ["intern"]) == 0
    assert await _count(pools, "globex", ["hr"]) == 1


async def test_context_does_not_leak_to_next_borrower(pools: Pools, seeded: object) -> None:
    # Force reuse: drain the pool to one connection's worth by holding the others.
    async with scoped(pools.app, "acme", ["hr"], acquire_timeout_s=1) as c:
        pid = await c.fetchval("SELECT pg_backend_pid()")
        assert await c.fetchval("SELECT count(*) FROM core.chunks") == 2
    held = []
    try:
        while True:
            conn = await pools.app.acquire(timeout=0.5)
            held.append(conn)
            if await conn.fetchval("SELECT pg_backend_pid()") == pid:
                roles = await conn.fetchval("SELECT current_setting('app.roles', true)")
                assert roles in ("", None)
                assert await conn.fetchval("SELECT count(*) FROM core.chunks") == 0
                break
    finally:
        for conn in held:
            await pools.app.release(conn)


async def test_session_level_set_is_reset_on_release(pools: Pools, seeded: object) -> None:
    # Even a careless session-level SET (not SET LOCAL) must not survive release.
    # asyncpg runs RESET ALL on release; this pins that behavior so an upgrade can't drop it.
    async with pools.app.acquire() as c:
        pid = await c.fetchval("SELECT pg_backend_pid()")
        await c.execute("SELECT set_config('app.tenant', 'acme', false)")
        await c.execute("SELECT set_config('app.roles', '{hr}', false)")
    held = []
    try:
        while True:
            conn = await pools.app.acquire(timeout=0.5)
            held.append(conn)
            if await conn.fetchval("SELECT pg_backend_pid()") == pid:
                assert await conn.fetchval("SELECT count(*) FROM core.chunks") == 0
                break
    finally:
        for conn in held:
            await pools.app.release(conn)


@pytest.mark.parametrize("bad", ["hr}", "a,b", 'x"', "a b", "{hr}", "", "A", "x" * 65])
def test_role_literal_rejects_injection(bad: str) -> None:
    with pytest.raises(ValueError):
        roles_literal([bad])


async def test_scoped_rejects_bad_tenant(pools: Pools) -> None:
    with pytest.raises(ValueError):
        async with scoped(pools.app, "acme' OR '1'='1", ["hr"], acquire_timeout_s=1):
            pass
