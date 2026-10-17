"""Envelope encryption: a master key (env) wraps per-tenant keys (DB).

Why per-tenant keys:
  - The HMAC key decides tokens. Separate keys mean the same name gets unrelated tokens in
    two tenants, so tokens can't be used to link people across customers.
  - Deleting a tenant's encryption keys makes all its vault ciphertext unreadable at once
    (crypto-shredding), even in backups.
Why wrap them with a master key:
  - Rotating the master key means re-wrapping a few rows per tenant, not re-encrypting
    every vault row.
  - A DB dump alone (no env) can't decrypt anything.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import asyncpg
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from govrag.errors import IntegrityError, Unavailable

HMAC_KEY_BYTES = 32


class MasterKey:
    def __init__(self, keys_csv: str) -> None:
        keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        if not keys:
            raise ValueError("GOVRAG_MASTER_KEYS is empty")
        # MultiFernet: encrypts with keys[0], decrypts with any. That's what makes
        # master key rotation a non-event: add the new key first, re-wrap, drop the old one.
        self._mf = MultiFernet([Fernet(k.encode()) for k in keys])

    def wrap(self, raw: bytes) -> bytes:
        return self._mf.encrypt(raw)

    def unwrap(self, wrapped: bytes) -> bytes:
        try:
            return self._mf.decrypt(wrapped)
        except InvalidToken as e:
            # Wrong or missing master key. Never fall back to anything.
            raise IntegrityError("master_key_mismatch") from e

    def rewrap(self, wrapped: bytes) -> bytes:
        return self._mf.rotate(wrapped)


@dataclass(frozen=True, slots=True)
class EncKeys:
    fernet: MultiFernet  # newest version first: encrypts with newest, decrypts with all
    current_version: int


class TenantKeyring:
    """Caches unwrapped tenant keys in memory for ttl_s.

    The cache only saves a DB round trip and an unwrap. Shredding deletes the vault rows
    in the same transaction as the keys, so a stale cached key has nothing to decrypt.
    """

    def __init__(self, master: MasterKey, ttl_s: float = 60.0) -> None:
        self._master = master
        self._ttl = ttl_s
        self._hmac: dict[str, tuple[float, bytes]] = {}
        self._enc: dict[str, tuple[float, EncKeys]] = {}

    def invalidate(self, tenant_id: str) -> None:
        self._hmac.pop(tenant_id, None)
        self._enc.pop(tenant_id, None)

    def cached_hmac(self, tenant_id: str) -> bytes | None:
        """The hot path: no DB connection needed while the key is cached."""
        hit = self._hmac.get(tenant_id)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit[1]
        return None

    async def hmac_key(self, conn: asyncpg.Connection, tenant_id: str) -> bytes:
        """conn must be scoped() to tenant_id (RLS shows only that tenant's row)."""
        hit = self._hmac.get(tenant_id)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit[1]
        wrapped = await conn.fetchval(
            "SELECT wrapped_key FROM core.tenant_hmac_keys WHERE tenant_id = $1", tenant_id
        )
        if wrapped is None:
            raise Unavailable("tenant_not_provisioned")
        key = self._master.unwrap(wrapped)
        self._hmac[tenant_id] = (time.monotonic(), key)
        return key

    async def enc_keys(self, conn: asyncpg.Connection, tenant_id: str) -> EncKeys:
        hit = self._enc.get(tenant_id)
        if hit and time.monotonic() - hit[0] < self._ttl:
            return hit[1]
        rows = await conn.fetch(
            "SELECT key_version, wrapped_key FROM vault.tenant_enc_keys"
            " WHERE tenant_id = $1 ORDER BY key_version DESC",
            tenant_id,
        )
        if not rows:
            raise Unavailable("tenant_keys_missing")
        keys = EncKeys(
            fernet=MultiFernet([Fernet(self._master.unwrap(r["wrapped_key"])) for r in rows]),
            current_version=rows[0]["key_version"],
        )
        self._enc[tenant_id] = (time.monotonic(), keys)
        return keys


async def provision_tenant(owner: asyncpg.Connection, master: MasterKey, tenant_id: str) -> None:
    """Create a tenant with fresh keys. Owner login only (an ops action, not an API)."""
    async with owner.transaction():
        await owner.execute("INSERT INTO core.tenants (id) VALUES ($1)", tenant_id)
        await owner.execute(
            "INSERT INTO core.tenant_hmac_keys (tenant_id, wrapped_key) VALUES ($1, $2)",
            tenant_id,
            master.wrap(os.urandom(HMAC_KEY_BYTES)),
        )
        await owner.execute(
            "INSERT INTO vault.tenant_enc_keys (tenant_id, key_version, wrapped_key)"
            " VALUES ($1, 1, $2)",
            tenant_id,
            master.wrap(Fernet.generate_key()),
        )


async def rotate_master(owner: asyncpg.Connection, master: MasterKey) -> int:
    """Re-wrap every tenant key with the current primary master key. Returns rows touched."""
    n = 0
    async with owner.transaction():
        for table in ("core.tenant_hmac_keys", "vault.tenant_enc_keys"):
            pk = "tenant_id" if table.startswith("core") else "tenant_id, key_version"
            rows = await owner.fetch(f"SELECT {pk}, wrapped_key FROM {table} FOR UPDATE")  # noqa: S608
            for r in rows:
                new = master.rewrap(r["wrapped_key"])
                if table.startswith("core"):
                    await owner.execute(
                        f"UPDATE {table} SET wrapped_key = $1 WHERE tenant_id = $2",  # noqa: S608
                        new,
                        r["tenant_id"],
                    )
                else:
                    await owner.execute(
                        f"UPDATE {table} SET wrapped_key = $1"  # noqa: S608
                        " WHERE tenant_id = $2 AND key_version = $3",
                        new,
                        r["tenant_id"],
                        r["key_version"],
                    )
                n += 1
    return n
