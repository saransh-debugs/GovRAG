"""Per-tenant policy cache. Every request needs the policy; a DB read per request would
double the query path's DB round trips.

Freshness, two ways on purpose:
  1. LISTEN policy_changed: an edit invalidates the tenant's entry on every instance
     within milliseconds of COMMIT.
  2. TTL (30 s): if a notification is lost (listener connection dropped, network
     partition), the entry still expires. So a removed permission lasts at most ttl_s.
On listener reconnect the whole cache is cleared: notifications sent while we were
disconnected are gone for good, so any entry might be stale.

Fail closed: entry expired + DB unreachable = Unavailable. We never serve a stale
policy past its TTL, and never fall back to "no policy = allow".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass

import asyncpg

from govrag.db import Pools, scoped
from govrag.errors import GovRagError, Unavailable
from govrag.policy.model import ActivePolicy
from govrag.policy.store import read_active

log = logging.getLogger(__name__)
CHANNEL = "policy_changed"


@dataclass(slots=True)
class _Entry:
    fetched_at: float
    value: ActivePolicy


class PolicyCache:
    def __init__(self, pools: Pools, listen_dsn: str, *, ttl_s: float = 30.0) -> None:
        self._pools = pools
        self._dsn = listen_dsn
        self._ttl = ttl_s
        self._entries: dict[str, _Entry] = {}
        # Single-flight: 500 requests missing the same tenant at once -> 1 DB read, not 500.
        self._inflight: dict[str, asyncio.Future[ActivePolicy]] = {}
        self._listener: asyncpg.Connection | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self.listening = asyncio.Event()
        self.stats = {"hits": 0, "misses": 0, "notifies": 0, "reconnects": 0}

    # --- reads -------------------------------------------------------------------------

    async def get(self, tenant_id: str) -> ActivePolicy:
        e = self._entries.get(tenant_id)
        if e is not None and time.monotonic() - e.fetched_at < self._ttl:
            self.stats["hits"] += 1
            return e.value
        self.stats["misses"] += 1
        fut = self._inflight.get(tenant_id)
        if fut is not None:
            return await asyncio.shield(fut)
        fut = asyncio.get_running_loop().create_future()
        self._inflight[tenant_id] = fut
        try:
            value = await self._fetch(tenant_id)
        except BaseException as err:
            fut.set_exception(err)
            # Mark the exception retrieved so a future nobody else awaited doesn't log.
            fut.exception()
            raise
        else:
            self._entries[tenant_id] = _Entry(time.monotonic(), value)
            fut.set_result(value)
            return value
        finally:
            self._inflight.pop(tenant_id, None)

    async def _fetch(self, tenant_id: str) -> ActivePolicy:
        try:
            async with scoped(
                self._pools.app, tenant_id, acquire_timeout_s=self._pools.acquire_timeout_s
            ) as c:
                return await read_active(c, tenant_id)
        except GovRagError:
            raise
        except (OSError, asyncpg.PostgresError) as e:
            raise Unavailable("policy_unavailable") from e

    def invalidate(self, tenant_id: str | None = None) -> None:
        if tenant_id is None:
            self._entries.clear()
        else:
            self._entries.pop(tenant_id, None)

    # --- listener ------------------------------------------------------------------------

    async def start(self) -> None:
        self._task = asyncio.create_task(self._listen_forever(), name="policy-listener")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._listener and not self._listener.is_closed():
            await self._listener.close()

    def _on_notify(self, _conn: object, _pid: int, _channel: str, payload: str) -> None:
        self.stats["notifies"] += 1
        self.invalidate(payload)

    async def _listen_forever(self) -> None:
        backoff = 0.2
        while not self._stopping:
            lost = asyncio.Event()
            try:
                # A dedicated connection, not a pooled one: LISTEN state lives on the
                # session, and a pooled connection would drop it on release.
                conn = await asyncpg.connect(self._dsn, timeout=5)
                conn.add_termination_listener(lambda _c, ev=lost: ev.set())
                await conn.add_listener(CHANNEL, self._on_notify)
                self._listener = conn
                # Anything sent while we were disconnected is lost; start clean.
                self.invalidate()
                self.listening.set()
                backoff = 0.2
                # A silent network drop never fires the termination listener. Ping, so a
                # dead listener is noticed in ~10 s instead of never.
                while not lost.is_set():
                    try:
                        await asyncio.wait_for(lost.wait(), timeout=10.0)
                    except TimeoutError:
                        try:
                            await conn.fetchval("SELECT 1", timeout=3.0)
                        except (OSError, asyncpg.PostgresError, TimeoutError,
                                asyncpg.InterfaceError):
                            break
                if not conn.is_closed():
                    conn.terminate()
            except asyncio.CancelledError:
                raise
            except (OSError, asyncpg.PostgresError, TimeoutError) as e:
                log.warning("policy listener connect failed: %s", type(e).__name__)
            finally:
                self.listening.clear()
            self.stats["reconnects"] += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)
