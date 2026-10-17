"""Admission control: refuse work early instead of queueing it without bound.

Why not just await a semaphore: a request that waits in a queue still holds its socket,
its parsed body and its coroutine. Under a burst, the queue grows until memory runs out,
and every queued request times out anyway. Rejecting with 503 + Retry-After at the door
costs microseconds and keeps latency flat for the requests we do accept.

Per-tenant caps stop one noisy tenant from taking every global slot (noisy neighbor).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

from govrag.errors import Overloaded


class Admission:
    def __init__(self, *, global_max: int, per_tenant_max: int) -> None:
        self._global_max = global_max
        self._tenant_max = per_tenant_max
        self._inflight = 0
        self._by_tenant: Counter[str] = Counter()
        self.rejected: Counter[str] = Counter()

    @contextmanager
    def admit(self, tenant_id: str) -> Iterator[None]:
        # No await between check and increment, so this is atomic on the event loop.
        if self._by_tenant[tenant_id] >= self._tenant_max:
            self.rejected["tenant"] += 1
            raise Overloaded("tenant_busy", retry_after_s=1.0)
        if self._inflight >= self._global_max:
            self.rejected["global"] += 1
            raise Overloaded("server_busy", retry_after_s=1.0)
        self._inflight += 1
        self._by_tenant[tenant_id] += 1
        try:
            yield
        finally:
            self._inflight -= 1
            self._by_tenant[tenant_id] -= 1
            if self._by_tenant[tenant_id] == 0:
                del self._by_tenant[tenant_id]  # don't grow one entry per tenant forever

    @property
    def inflight(self) -> int:
        return self._inflight
