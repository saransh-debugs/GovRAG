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
import time
from dataclasses import dataclass

import asyncpg

from govrag.db import Pools, scoped
from govrag.errors import GovRagError, Unavailable
from govrag.notify import Listener
from govrag.policy.model import ActivePolicy
from govrag.policy.store import read_active

CHANNEL = "policy_changed"


@dataclass(slots=True)
class _Entry:
    fetched_at: float
    value: ActivePolicy


class PolicyCache:
    def __init__(self, pools: Pools, listener: Listener | str, *, ttl_s: float = 30.0) -> None:
        """listener: a shared Listener, or a DSN to own a private one (tests)."""
        self._pools = pools
        self._own = isinstance(listener, str)
        self._listener = Listener(listener) if isinstance(listener, str) else listener
        self._listener.subscribe(CHANNEL, self._on_notify, on_connect=self._on_connect)
        self._ttl = ttl_s
        self._entries: dict[str, _Entry] = {}
        # Single-flight: 500 requests missing the same tenant at once -> 1 DB read, not 500.
        self._inflight: dict[str, asyncio.Future[ActivePolicy]] = {}
        self.stats = {"hits": 0, "misses": 0, "notifies": 0, "reconnects": 0}

    @property
    def listening(self) -> asyncio.Event:
        return self._listener.listening

    @property
    def connection(self) -> asyncpg.Connection | None:
        return self._listener.connection

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

    # --- push invalidation -------------------------------------------------------------

    async def start(self) -> None:
        if self._own:
            await self._listener.start()

    async def stop(self) -> None:
        if self._own:
            await self._listener.stop()

    def _on_notify(self, _conn: object, _pid: int, _channel: str, payload: str) -> None:
        self.stats["notifies"] += 1
        self.invalidate(payload)

    def _on_connect(self) -> None:
        # Anything sent while we were disconnected is lost; start clean.
        self.invalidate()
        self.stats["reconnects"] = self._listener.connects - 1
