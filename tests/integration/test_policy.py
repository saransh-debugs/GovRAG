"""Policies: validation, permissions, versioning, cache freshness, re-scan."""

from __future__ import annotations

import asyncio
import os
import re
import time

import asyncpg
import pytest
from pydantic import ValidationError

from govrag.auth import Principal
from govrag.chunking import Chunker
from govrag.config import Settings
from govrag.db import Pools, scoped
from govrag.embed import HashEmbedder
from govrag.errors import Conflict, Forbidden, NotFound, Unavailable
from govrag.ingest import DocumentIn, Ingestor, SectionIn
from govrag.keys import MasterKey, TenantKeyring, provision_tenant
from govrag.pii.analyzer import PiiAnalyzer
from govrag.pii.tokens import TOKEN_RE
from govrag.policy.cache import PolicyCache
from govrag.policy.model import Policy
from govrag.policy.rescan import Rescanner
from govrag.policy.store import PolicyStore


@pytest.fixture
async def tenant(owner: asyncpg.Connection, master: MasterKey) -> str:
    t = f"t-{os.urandom(4).hex()}"
    await provision_tenant(owner, master, t)
    return t


def admin(t: str) -> Principal:
    return Principal(t, "admin1", frozenset({"policy_admin"}))


def listen_dsn(s: Settings) -> str:
    return s.dsn("govrag_app", s.db_app_password)


# --- model -------------------------------------------------------------------------------

@pytest.mark.parametrize("doc", [
    {"unmask": {"hr": ["NOT_A_TYPE"]}},
    {"unmask": {"HR!": ["PERSON"]}},
    {"extra_redact": ["PERSON"]},            # already default, not optional
    {"llm": {"models": ["gpt-evil"]}},       # not on the server allowlist
    {"llm": {"models": []}},
    {"llm": {"region": "mars"}},
    {"surprise": 1},                         # unknown field
])
def test_bad_policies_rejected(doc: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Policy.model_validate(doc)


def test_may_unmask_is_union_of_roles() -> None:
    p = Policy.model_validate(
        {"unmask": {"hr": ["PERSON", "US_SSN"], "support": ["EMAIL_ADDRESS"]}}
    )
    assert p.may_unmask(frozenset({"hr", "support"})) == {"PERSON", "US_SSN", "EMAIL_ADDRESS"}
    assert p.may_unmask(frozenset({"intern"})) == frozenset()


# --- store -------------------------------------------------------------------------------

async def test_only_admin_writes_and_db_checks_again(pools: Pools, tenant: str) -> None:
    store = PolicyStore(pools)
    with pytest.raises(Forbidden):
        await store.put(Principal(tenant, "u", frozenset({"hr"})), Policy(), expected_version=0)
    # Bypass the Python check: the DB's RLS policy still refuses a non-admin write.
    async with scoped(pools.app, tenant, ["hr"], acquire_timeout_s=1) as c:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await c.execute(
                "INSERT INTO policy.versions (tenant_id, version, doc, created_by)"
                " VALUES ($1, 1, '{}', 'x')", tenant,
            )


async def test_admin_of_one_tenant_cannot_write_another(pools: Pools, tenant: str) -> None:
    async with scoped(pools.app, tenant, ["policy_admin"], acquire_timeout_s=1) as c:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await c.execute(
                "INSERT INTO policy.versions (tenant_id, version, doc, created_by)"
                " VALUES ('globex', 99, '{}', 'x')"
            )


async def test_versions_conflict_and_rollback(
    pools: Pools, tenant: str, owner: asyncpg.Connection
) -> None:
    store = PolicyStore(pools)
    a = admin(tenant)
    assert (await store.get(a)).version == 0
    v1 = await store.put(a, Policy.model_validate({"unmask": {"hr": ["PERSON"]}}),
                         expected_version=0)
    v2 = await store.put(a, Policy.model_validate({"unmask": {"hr": ["PERSON", "US_SSN"]}}),
                         expected_version=1)
    assert (v1.version, v2.version) == (1, 2)
    with pytest.raises(Conflict):  # someone still editing on top of v1
        await store.put(a, Policy(), expected_version=1)
    back = await store.rollback(a, 1, expected_version=2)
    assert back.version == 1 and back.policy.may_unmask(frozenset({"hr"})) == {"PERSON"}
    assert (await store.get(a)).version == 1
    with pytest.raises(NotFound):
        await store.rollback(a, 42, expected_version=1)
    # Versions are immutable, even for the owner.
    with pytest.raises(asyncpg.InsufficientPrivilegeError, match="immutable"):
        await owner.execute("UPDATE policy.versions SET doc = '{}' WHERE tenant_id = $1", tenant)
    actions = await owner.fetch(
        "SELECT action, policy_version, prev_policy_version FROM audit.events"
        " WHERE tenant_id = $1 ORDER BY id", tenant,
    )
    assert [tuple(r) for r in actions] == [
        ("policy_change", 1, 0), ("policy_change", 2, 1), ("policy_rollback", 1, 2),
    ]


async def test_concurrent_first_saves_one_wins(pools: Pools, tenant: str) -> None:
    store = PolicyStore(pools)
    res = await asyncio.gather(
        *(store.put(admin(tenant), Policy(), expected_version=0) for _ in range(5)),
        return_exceptions=True,
    )
    assert sum(not isinstance(r, BaseException) for r in res) == 1
    assert all(isinstance(r, Conflict) for r in res if isinstance(r, BaseException))


# --- cache -------------------------------------------------------------------------------

async def _wait_for(cond: object, within_s: float) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < within_s:
        if await cond():  # type: ignore[operator]
            return time.perf_counter() - t0
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")


async def test_cache_notify_path_under_a_second(
    pools: Pools, tenant: str, settings: Settings
) -> None:
    cache = PolicyCache(pools, listen_dsn(settings), ttl_s=300)  # TTL can't be what saves us
    await cache.start()
    try:
        await asyncio.wait_for(cache.listening.wait(), 5)
        assert (await cache.get(tenant)).version == 0
        await PolicyStore(pools).put(admin(tenant), Policy(), expected_version=0)

        async def updated() -> bool:
            return (await cache.get(tenant)).version == 1

        took = await _wait_for(updated, 1.0)
        assert cache.stats["notifies"] >= 1
        print(f"notify path: {took * 1000:.0f} ms")
    finally:
        await cache.stop()


async def test_cache_ttl_path_when_notify_lost(
    pools: Pools, tenant: str, settings: Settings
) -> None:
    cache = PolicyCache(pools, listen_dsn(settings), ttl_s=1.0)
    await cache.start()
    try:
        await asyncio.wait_for(cache.listening.wait(), 5)
        assert (await cache.get(tenant)).version == 0
        # Lose the notification: unhook the callback without dropping the connection.
        assert cache.connection is not None
        await cache.connection.remove_listener("policy_changed", cache._on_notify)
        await PolicyStore(pools).put(admin(tenant), Policy(), expected_version=0)
        assert (await cache.get(tenant)).version == 0  # stale, within TTL

        async def updated() -> bool:
            return (await cache.get(tenant)).version == 1

        took = await _wait_for(updated, 2.0)
        assert took <= 1.2
    finally:
        await cache.stop()


async def test_cache_listener_reconnects_and_clears(
    pools: Pools, tenant: str, settings: Settings, owner: asyncpg.Connection
) -> None:
    cache = PolicyCache(pools, listen_dsn(settings), ttl_s=300)
    await cache.start()
    try:
        await asyncio.wait_for(cache.listening.wait(), 5)
        await cache.get(tenant)
        assert cache.connection is not None
        pid = cache.connection.get_server_pid()
        await owner.execute("SELECT pg_terminate_backend($1)", pid)

        async def reconnected() -> bool:
            lst = cache.connection
            return cache.listening.is_set() and lst is not None and lst.get_server_pid() != pid

        await _wait_for(reconnected, 5.0)
        assert cache.stats["reconnects"] >= 1
        assert tenant not in cache._entries  # cleared: notifications may have been missed
    finally:
        await cache.stop()


async def test_cache_fails_closed_when_db_down(settings: Settings, tenant: str) -> None:
    dead = await Pools.create(settings)
    await dead.close()
    cache = PolicyCache(dead, listen_dsn(settings))
    with pytest.raises(Unavailable):
        await cache.get(tenant)


async def test_cache_single_flight(pools: Pools, tenant: str, settings: Settings) -> None:
    cache = PolicyCache(pools, listen_dsn(settings))
    res = await asyncio.gather(*(cache.get(tenant) for _ in range(200)))
    assert {r.version for r in res} == {0}
    assert cache.stats["misses"] == 200 and len(cache._entries) == 1


# --- re-scan -----------------------------------------------------------------------------

async def test_rescan_resumes_without_double_work(
    pools: Pools, tenant: str, settings: Settings, analyzer: PiiAnalyzer,
    master: MasterKey, owner: asyncpg.Connection,
) -> None:
    keyring = TenantKeyring(master)
    ing = Ingestor(pools=pools, keyring=keyring, analyzer=analyzer, embedder=HashEmbedder(),
                   chunker=Chunker(), settings=settings)
    svc = Principal(tenant, "svc", frozenset({"ingest"}))
    cities = ["Denver", "Boston", "Seattle"]
    for c in cities:
        await ing.ingest(svc, DocumentIn("Trip", ("hr",), (
            SectionIn(f"Margaret Okafor flew to {c} for the board meeting."),
        )))
    before = await owner.fetch("SELECT text_redacted FROM core.chunks WHERE tenant_id = $1", tenant)
    assert all(any(c in r["text_redacted"] for c in cities) for r in before)

    await PolicyStore(pools).put(
        admin(tenant), Policy.model_validate({"extra_redact": ["LOCATION"]}), expected_version=0
    )
    rs = Rescanner(pools=pools, keyring=keyring, analyzer=analyzer, embedder=HashEmbedder(),
                   pause_s=0)
    job = await rs.claim()
    while job is not None and job.tenant_id != tenant:  # other tests may have left jobs
        await rs.run_job(job)
        job = await rs.claim()
    assert job is not None
    assert await rs.run_job(job, max_docs=1) is False  # "crash" after one document
    # Another worker can't steal it while the lease is valid...
    other = await rs.claim()
    assert other is None or other.tenant_id != tenant
    # ...but once the lease runs out, it can.
    await owner.execute("UPDATE core.rescan_jobs SET lease_until = now() - interval '1 s'"
                        " WHERE tenant_id = $1", tenant)
    again = await rs.claim()
    assert again == job
    assert await rs.run_job(again) is True

    after = await owner.fetch(
        "SELECT text_redacted, redaction_rev, pii_types FROM core.chunks WHERE tenant_id = $1",
        tenant,
    )
    for r in after:
        assert not any(c in r["text_redacted"] for c in cities)
        assert r["redaction_rev"] == 1 and "LOCATION" in r["pii_types"]
        # No nested tokens: every "[" starts exactly one well-formed token.
        assert r["text_redacted"].count("[") == len(TOKEN_RE.findall(r["text_redacted"]))
        assert not re.search(r"\[[A-Z_]+_\[", r["text_redacted"])
    job_row = await owner.fetchrow(
        "SELECT status, docs_done FROM core.rescan_jobs WHERE tenant_id = $1", tenant
    )
    assert job_row["status"] == "done" and job_row["docs_done"] == len(cities)
