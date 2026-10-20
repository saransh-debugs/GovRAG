"""Revoked tokens stop working everywhere within milliseconds, not at exp."""

from __future__ import annotations

import asyncio
import os
import time

import asyncpg
import pytest

from govrag.auth import Principal
from govrag.config import Settings
from govrag.db import Pools
from govrag.errors import Forbidden, Unauthorized, Unavailable
from govrag.keys import MasterKey, provision_tenant
from govrag.notify import Listener
from govrag.revocations import Revocations


@pytest.fixture
async def tenant(owner: asyncpg.Connection, master: MasterKey) -> str:
    t = f"r-{os.urandom(3).hex()}"
    await provision_tenant(owner, master, t)
    return t


def dsn(s: Settings) -> str:
    return s.dsn("govrag_app", s.db_app_password)


async def test_revoke_rejects_old_tokens_on_every_instance(
    pools: Pools, settings: Settings, tenant: str
) -> None:
    la, lb = Listener(dsn(settings)), Listener(dsn(settings))
    a = Revocations(pools, la, max_token_age_s=3600, ttl_s=300)  # TTL can't save us
    b = Revocations(pools, lb, max_token_age_s=3600, ttl_s=300)
    await la.start()
    await lb.start()
    try:
        await asyncio.wait_for(asyncio.gather(la.listening.wait(), lb.listening.wait()), 5)
        old = Principal(tenant, "fired-1", frozenset({"hr"}), issued_at=time.time() - 60)
        await a.check(old)
        await b.check(old)  # both have loaded the (empty) list
        with pytest.raises(Forbidden):
            await a.revoke(Principal(tenant, "u", frozenset({"hr"})), "fired-1")
        await a.revoke(Principal(tenant, "admin", frozenset({"policy_admin"})), "fired-1")
        with pytest.raises(Unauthorized, match="token_revoked"):
            await a.check(old)  # the instance that revoked: immediately
        t0 = time.perf_counter()
        while True:  # the other instance: via NOTIFY
            try:
                await b.check(old)
            except Unauthorized:
                break
            assert time.perf_counter() - t0 < 1.0, "NOTIFY didn't arrive"
            await asyncio.sleep(0.01)
        # A token issued after the revocation (the user logged in again) works.
        await b.check(Principal(tenant, "fired-1", frozenset({"hr"}), issued_at=time.time() + 1))
        # Other users and other tenants are untouched.
        await b.check(Principal(tenant, "someone-else", frozenset(), issued_at=0))
        await b.check(Principal("globex", "fired-1", frozenset(), issued_at=0))
    finally:
        await la.stop()
        await lb.stop()


async def test_stale_list_and_db_down_fails_closed(settings: Settings) -> None:
    dead = await Pools.create(settings)
    await dead.close()
    r = Revocations(dead, Listener(dsn(settings)), max_token_age_s=3600)
    with pytest.raises(Unavailable, match="revocations_unavailable"):
        await r.check(Principal("acme", "u", frozenset(), issued_at=time.time()))
