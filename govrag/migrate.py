"""Tiny migration runner: applies migrations/NNN_*.sql in order, once each.

Run as the owner login. Each file runs in its own transaction, so a broken migration
leaves the schema at the previous version, not half-applied.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import asyncpg

from govrag.config import Settings, get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if _NAME.match(p.name))
    numbers = [p.name[:3] for p in files]
    if len(set(numbers)) != len(numbers):
        raise RuntimeError(f"duplicate migration numbers in {directory}")
    return files


async def apply(conn: asyncpg.Connection, settings: Settings) -> list[str]:
    await conn.execute(
        "CREATE SCHEMA IF NOT EXISTS meta;"
        "CREATE TABLE IF NOT EXISTS meta.schema_migrations ("
        "  name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    done = {r["name"] for r in await conn.fetch("SELECT name FROM meta.schema_migrations")}
    applied: list[str] = []
    for path in migration_files():
        if path.name in done:
            continue
        async with conn.transaction():
            await conn.execute(path.read_text())
            await conn.execute("INSERT INTO meta.schema_migrations (name) VALUES ($1)", path.name)
        applied.append(path.name)

    # Passwords come from env. ALTER ROLE can't take bind parameters, so build it with
    # format(%I, %L), which quotes safely inside Postgres.
    for role, secret in (
        ("govrag_app", settings.db_app_password),
        ("govrag_vault", settings.db_vault_password),
        ("govrag_ingest", settings.db_ingest_password),
    ):
        stmt = await conn.fetchval(
            "SELECT format('ALTER ROLE %I PASSWORD %L', $1::text, $2::text)",
            role,
            secret.get_secret_value(),
        )
        await conn.execute(stmt)
    return applied


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    s = get_settings()
    conn = await asyncpg.connect(s.dsn(s.db_owner_user, s.db_owner_password))
    try:
        for name in await apply(conn, s):
            print(f"applied {name}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(_main())
