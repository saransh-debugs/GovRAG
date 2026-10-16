"""Connection pools, one per DB login, and the tenant/role-scoped transaction helper.

Every query in the app goes through `scoped()`. It sets the tenant and roles with
set_config(..., is_local=true), so they last for this transaction only. Without is_local,
the setting would stay on the pooled connection and the *next* request to borrow it would
run with this request's roles.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
from pgvector.asyncpg import register_vector

from govrag.config import Settings
from govrag.errors import Overloaded, Unavailable

# Roles and tenant IDs go into a Postgres array literal ("{a,b}"). This pattern has no
# comma, brace, quote, backslash or space, so a role can't break out of the literal.
# auth.py validates the same pattern; this checks it again right before the DB, on purpose.
IDENT = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
TENANT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def roles_literal(roles: Iterable[str]) -> str:
    out = sorted(set(roles))
    for r in out:
        if not IDENT.match(r):
            raise ValueError("invalid role identifier")
    return "{" + ",".join(out) + "}" if out else ""


async def _init_conn(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


@dataclass(slots=True)
class Pools:
    app: asyncpg.Pool
    vault: asyncpg.Pool
    ingest: asyncpg.Pool
    acquire_timeout_s: float

    @classmethod
    async def create(cls, s: Settings) -> Pools:
        async def make(user: str, password: object, size: int, name: str) -> asyncpg.Pool:
            pool = await asyncpg.create_pool(
                s.dsn(user, password),  # type: ignore[arg-type]
                min_size=1,
                max_size=size,
                init=_init_conn,
                # Client-side cap on any one call; the server-side statement_timeout is
                # the real guard (it frees the backend, not just our await).
                command_timeout=s.db_statement_timeout_ms / 1000 + 1,
                server_settings={
                    "statement_timeout": str(s.db_statement_timeout_ms),
                    "application_name": name,
                },
            )
            assert pool is not None
            return pool

        app = await make("govrag_app", s.db_app_password, s.db_app_pool_max, "govrag-app")
        vault = await make("govrag_vault", s.db_vault_password, s.db_vault_pool_max, "govrag-vault")
        ingest = await make(
            "govrag_ingest", s.db_ingest_password, s.db_ingest_pool_max, "govrag-ingest"
        )
        return cls(app=app, vault=vault, ingest=ingest, acquire_timeout_s=s.db_acquire_timeout_s)

    async def close(self) -> None:
        await asyncio.gather(self.app.close(), self.vault.close(), self.ingest.close())


@asynccontextmanager
async def scoped(
    pool: asyncpg.Pool,
    tenant_id: str,
    roles: Iterable[str] = (),
    *,
    acquire_timeout_s: float,
    extra: dict[str, str] | None = None,
) -> AsyncIterator[asyncpg.Connection]:
    """Borrow a connection, open a transaction, set tenant + roles for it only.

    Pool empty for acquire_timeout_s -> Overloaded (503 + Retry-After), not an unbounded
    wait. An unbounded wait turns a DB slowdown into a pile-up of hung requests that
    eventually exhausts memory.
    """
    if not TENANT.match(tenant_id):
        raise ValueError("invalid tenant id")
    lit = roles_literal(roles)
    try:
        conn = await pool.acquire(timeout=acquire_timeout_s)
    except TimeoutError as e:
        raise Overloaded("db_pool_exhausted", retry_after_s=1.0) from e
    except (OSError, asyncpg.PostgresConnectionError, asyncpg.InterfaceError) as e:
        raise Unavailable("db_unreachable") from e
    try:
        async with conn.transaction():
            names = ["app.tenant", "app.roles", *(extra or {})]
            values = [tenant_id, lit, *(extra or {}).values()]
            sets = ", ".join(
                f"set_config(${2 * i + 1}, ${2 * i + 2}, true)" for i in range(len(names))
            )
            args: list[str] = [x for pair in zip(names, values, strict=True) for x in pair]
            await conn.execute(f"SELECT {sets}", *args)
            yield conn
    except (OSError, asyncpg.PostgresConnectionError) as e:
        raise Unavailable("db_unreachable") from e
    finally:
        await pool.release(conn)
