"""Token vault: store and load encrypted PII values.

All writes and reads are batched into one round trip (unnest / ANY). A per-token
INSERT loop on a 500-chunk document is 500+ round trips holding a transaction open.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import asyncpg
from cryptography.fernet import InvalidToken

from govrag.errors import IntegrityError
from govrag.keys import EncKeys
from govrag.pii.redact import Found

# Fernet is ~10-20 us per value. Below this many values, a thread hop costs more than it saves.
_THREAD_THRESHOLD = 256


async def _crypto(
    fn: Callable[[bytes], bytes], items: list[bytes], *, threaded: bool
) -> list[bytes]:
    if threaded:
        return await asyncio.to_thread(lambda: [fn(x) for x in items])
    return [fn(x) for x in items]


async def store(
    conn: asyncpg.Connection, tenant_id: str, found: Iterable[Found], keys: EncKeys
) -> int:
    """Insert tokens that don't exist yet. conn: ingest login, scoped to tenant_id.

    Returns how many were new. Raises IntegrityError on a token collision.
    """
    items = list({f.token: f for f in found}.values())
    if not items:
        return 0
    cts = await _crypto(
        keys.fernet.encrypt,
        [f.value.encode() for f in items],
        threaded=len(items) > _THREAD_THRESHOLD,
    )
    status = await conn.execute(
        """
        INSERT INTO vault.pii_tokens (tenant_id, token, entity_type, value_mac, ciphertext,
                                      key_version)
        SELECT $1, t, e, m, c, $6
          FROM unnest($2::text[], $3::text[], $4::bytea[], $5::bytea[]) AS u(t, e, m, c)
        ON CONFLICT (tenant_id, token) DO NOTHING
        """,
        tenant_id,
        [f.token for f in items],
        [f.entity_type for f in items],
        [f.mac for f in items],
        cts,
        keys.current_version,
    )
    # Same token but a different full MAC = two different values hashed to the same
    # 64-bit token. Mapping both to one ciphertext would show person A's name in place of
    # person B's. Fail the ingest instead.
    rows = await conn.fetch(
        "SELECT token, value_mac FROM vault.pii_tokens WHERE tenant_id = $1 AND token = ANY($2)",
        tenant_id,
        [f.token for f in items],
    )
    stored = {r["token"]: bytes(r["value_mac"]) for r in rows}
    for f in items:
        if stored.get(f.token) != f.mac:
            raise IntegrityError("token_collision")
    return int(status.split()[-1])


async def add_refs(
    conn: asyncpg.Connection, tenant_id: str, refs: Sequence[tuple[str, uuid.UUID]]
) -> None:
    if not refs:
        return
    await conn.execute(
        """
        INSERT INTO vault.token_refs (tenant_id, token, chunk_id)
        SELECT $1, t, c FROM unnest($2::text[], $3::uuid[]) AS u(t, c)
        ON CONFLICT DO NOTHING
        """,
        tenant_id,
        [t for t, _ in refs],
        [c for _, c in refs],
    )


@dataclass(frozen=True, slots=True)
class Revealed:
    entity_type: str
    value: str


async def load(
    conn: asyncpg.Connection, tenant_id: str, tokens: Iterable[str], keys: EncKeys
) -> dict[str, Revealed]:
    """Decrypt the given tokens. conn: vault login, scoped to tenant_id.

    Missing tokens are simply absent from the result. A row whose key was shredded (or a
    corrupted row) is skipped, not raised: the answer keeps its placeholder.
    """
    wanted = sorted(set(tokens))
    if not wanted:
        return {}
    rows = await conn.fetch(
        "SELECT token, entity_type, ciphertext FROM vault.pii_tokens"
        " WHERE tenant_id = $1 AND token = ANY($2)",
        tenant_id,
        wanted,
    )
    out: dict[str, Revealed] = {}

    def _decrypt_all() -> None:
        for r in rows:
            try:
                out[r["token"]] = Revealed(
                    r["entity_type"], keys.fernet.decrypt(bytes(r["ciphertext"])).decode()
                )
            except InvalidToken:
                continue

    if len(rows) > _THREAD_THRESHOLD:
        await asyncio.to_thread(_decrypt_all)
    else:
        _decrypt_all()
    return out


async def delete_document(conn: asyncpg.Connection, tenant_id: str, document_id: uuid.UUID) -> int:
    """Delete a document, its chunks, refs, and any token no other chunk still uses.

    conn: ingest login, scoped, inside a transaction. Returns how many tokens were erased.
    """
    tokens = await conn.fetch(
        "SELECT DISTINCT r.token FROM vault.token_refs r"
        " JOIN core.chunks c ON c.tenant_id = r.tenant_id AND c.id = r.chunk_id"
        " WHERE c.tenant_id = $1 AND c.document_id = $2",
        tenant_id,
        document_id,
    )
    await conn.execute(
        "DELETE FROM core.documents WHERE tenant_id = $1 AND id = $2", tenant_id, document_id
    )  # cascades to chunks -> token_refs
    if not tokens:
        return 0
    status = await conn.execute(
        """
        DELETE FROM vault.pii_tokens t
         WHERE t.tenant_id = $1 AND t.token = ANY($2)
           AND NOT EXISTS (SELECT 1 FROM vault.token_refs r
                            WHERE r.tenant_id = t.tenant_id AND r.token = t.token)
        """,
        tenant_id,
        [r["token"] for r in tokens],
    )
    return int(status.split()[-1])


async def shred_tenant(
    conn: asyncpg.Connection, tenant_id: str, new_wrapped_key: bytes
) -> int:
    """Crypto-shred a tenant's PII: delete its vault rows and encryption keys, then start a
    fresh key for future ingests. conn: ingest login, scoped, inside a transaction.

    Chunks keep their placeholders forever. Old backups hold ciphertext but no key that
    opens it (keys are wrapped by the master key, and the wrapped copies are gone too).
    """
    prev = await conn.fetchval(
        "SELECT coalesce(max(key_version), 0) FROM vault.tenant_enc_keys WHERE tenant_id = $1",
        tenant_id,
    )
    status = await conn.execute("DELETE FROM vault.pii_tokens WHERE tenant_id = $1", tenant_id)
    await conn.execute("DELETE FROM vault.tenant_enc_keys WHERE tenant_id = $1", tenant_id)
    # A new version number, never reused: a stale cached key on another instance can't
    # be mistaken for the new one.
    await conn.execute(
        "INSERT INTO vault.tenant_enc_keys (tenant_id, key_version, wrapped_key)"
        " VALUES ($1, $2, $3)",
        tenant_id,
        prev + 1,
        new_wrapped_key,
    )
    return int(status.split()[-1])
