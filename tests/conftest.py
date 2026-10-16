from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import asyncpg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pgvector.asyncpg import register_vector

from govrag import migrate
from govrag.auth import JwtVerifier, mint
from govrag.config import Settings
from govrag.db import Pools
from govrag.keys import MasterKey, TenantKeyring, provision_tenant

TEST_DB = "govrag_test"
TENANTS = ("acme", "globex")
ISS, AUD = "https://idp.test", "govrag-test"


@pytest.fixture(scope="session")
def settings() -> Settings:
    s = Settings()  # type: ignore[call-arg]  # secrets from .env
    s.db_name = TEST_DB
    s.db_acquire_timeout_s = 0.5
    return s


@pytest.fixture(scope="session")
async def owner(settings: Settings) -> AsyncIterator[asyncpg.Connection]:
    admin = await asyncpg.connect(
        settings.dsn(settings.db_owner_user, settings.db_owner_password).replace(
            f"/{TEST_DB}", "/postgres"
        )
    )
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        await admin.execute(f"CREATE DATABASE {TEST_DB}")
    finally:
        await admin.close()
    conn = await asyncpg.connect(settings.dsn(settings.db_owner_user, settings.db_owner_password))
    await migrate.apply(conn, settings)
    await register_vector(conn)
    master = MasterKey(settings.master_keys.get_secret_value())
    for t in TENANTS:
        await provision_tenant(conn, master, t)
    yield conn
    await conn.close()


@pytest.fixture(scope="session")
async def pools(settings: Settings, owner: asyncpg.Connection) -> AsyncIterator[Pools]:
    p = await Pools.create(settings)
    yield p
    await p.close()


@pytest.fixture(scope="session")
def master(settings: Settings) -> MasterKey:
    return MasterKey(settings.master_keys.get_secret_value())


@pytest.fixture
def keyring(master: MasterKey) -> TenantKeyring:
    return TenantKeyring(master, ttl_s=60)


@dataclass(frozen=True)
class JwtKit:
    private_pem: bytes
    public_pem: bytes
    verifier: JwtVerifier

    def token(self, tenant: str = "acme", user: str = "u1", roles: list[str] | None = None,
              **kw: object) -> str:
        return mint(
            self.private_pem,
            tenant_id=tenant,
            user_id=user,
            roles=roles if roles is not None else ["intern"],
            issuer=ISS,
            audience=AUD,
            **kw,  # type: ignore[arg-type]
        )


@pytest.fixture(scope="session")
def jwtkit() -> JwtKit:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return JwtKit(priv, pub, JwtVerifier(pub, issuer=ISS, audience=AUD, max_lifetime_s=3600))


async def insert_doc(
    owner: asyncpg.Connection,
    tenant: str,
    doc_roles: list[str],
    chunks: list[tuple[str, list[str]]],
    *,
    dim: int = 384,
) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """Owner-side seeding helper (owner is superuser, so RLS doesn't apply)."""
    import numpy as np

    doc = uuid.uuid4()
    await owner.execute(
        "INSERT INTO core.documents (tenant_id, id, title, allowed_roles, created_by)"
        " VALUES ($1, $2, 'seed', $3, 'seed')",
        tenant, doc, doc_roles,
    )
    ids = []
    rng = np.random.default_rng(abs(hash(doc)) % 2**32)
    for i, (text, roles) in enumerate(chunks):
        cid = uuid.uuid4()
        v = rng.standard_normal(dim).astype(np.float32)
        await owner.execute(
            "INSERT INTO core.chunks (tenant_id, id, document_id, ord, text_redacted, embedding,"
            " allowed_roles) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            tenant, cid, doc, i, text, v / np.linalg.norm(v), roles,
        )
        ids.append(cid)
    return doc, ids
