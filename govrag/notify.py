"""One LISTEN connection per process, shared by every cache that needs push invalidation.

A dedicated connection, not a pooled one: LISTEN state lives on the session, and a pooled
connection would drop it on release.

Failure handling, same for every subscriber:
  - termination listener: notices a closed connection right away;
  - heartbeat every 10 s: a silent network drop never fires the termination listener;
  - reconnect with backoff, then call every on_connect hook. Notifications sent while we
    were disconnected are gone for good, so subscribers must assume their cache is stale.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

import asyncpg

log = logging.getLogger(__name__)

Callback = Callable[[Any, int, str, str], None]  # asyncpg's (conn, pid, channel, payload)


class Listener:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._subs: list[tuple[str, Callback]] = []
        self._on_connect: list[Callable[[], None]] = []
        self.connection: asyncpg.Connection | None = None
        self.listening = asyncio.Event()
        self.connects = 0
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    def subscribe(
        self, channel: str, callback: Callback, on_connect: Callable[[], None] | None = None
    ) -> None:
        self._subs.append((channel, callback))
        if on_connect is not None:
            self._on_connect.append(on_connect)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="pg-listener")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.connection and not self.connection.is_closed():
            await self.connection.close()

    async def _run(self) -> None:
        backoff = 0.2
        while not self._stopping:
            lost = asyncio.Event()
            try:
                conn = await asyncpg.connect(self._dsn, timeout=5)
                conn.add_termination_listener(lambda _c, ev=lost: ev.set())
                for channel, cb in self._subs:
                    await conn.add_listener(channel, cb)
                self.connection = conn
                self.connects += 1
                for hook in self._on_connect:
                    hook()
                self.listening.set()
                backoff = 0.2
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
                log.warning("listener connect failed: %s", type(e).__name__)
            finally:
                self.listening.clear()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)
