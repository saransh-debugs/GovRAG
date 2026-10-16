"""Ingest end to end with real Presidio and real Postgres."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest
from cryptography.fernet import Fernet

from govrag.auth import Principal
from govrag.chunking import Chunker
from govrag.config import Settings
from govrag.db import Pools, scoped
from govrag.embed import HashEmbedder, ProcessEmbedder
from govrag.errors import BadRequest, Forbidden, IntegrityError, Overloaded
from govrag.ingest import DocumentIn, Ingestor, SectionIn
from govrag.keys import EncKeys, MasterKey, TenantKeyring, rotate_master
from govrag.pii import vault
from govrag.pii.analyzer import PiiAnalyzer
from govrag.pii.redact import Found
from govrag.pii.tokens import find_tokens

INGESTER = Principal("acme", "svc-ingest", frozenset({"ingest", "dpo"}))
GLOBEX_INGESTER = Principal("globex", "svc-ingest", frozenset({"ingest"}))

PII = {
    "names": ["Margaret Okafor", "Daniel Whitfield"],
    "ssn": "078-05-1120",
    "email": "m.okafor@acme-corp.com",
    "phone": "415-555-0132",
}
HR_TEXT = (
    f"Compensation review. {PII['names'][0]} (SSN {PII['ssn']}, {PII['email']}) gets a 12% raise. "
    f"Approved by {PII['names'][1]}; call {PII['phone']} with questions."
)
PUBLIC_TEXT = f"The offsite is in March. {PII['names'][1]} is organizing it."


@pytest.fixture(scope="session")
async def analyzer(settings: Settings) -> AsyncIterator[PiiAnalyzer]:
    a = PiiAnalyzer(model=settings.spacy_model, ingest_workers=1, query_workers=1)
    await a.start()
    yield a
    a.close()


def _ingestor(pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
              embedder: object | None = None) -> Ingestor:
    return Ingestor(pools=pools, keyring=keyring, analyzer=analyzer,
                    embedder=embedder or HashEmbedder(), chunker=Chunker(), settings=settings)


def _doc(title: str = "HR memo for Margaret Okafor") -> DocumentIn:
    return DocumentIn(
        title=title,
        allowed_roles=("hr", "intern"),
        sections=(SectionIn(HR_TEXT, ("hr",)), SectionIn(PUBLIC_TEXT)),
    )


async def _all_stored_text(owner: asyncpg.Connection) -> str:
    rows = await owner.fetch(
        "SELECT title AS t FROM core.documents UNION ALL SELECT text_redacted FROM core.chunks"
    )
    return "\n".join(r["t"] for r in rows)


async def test_ingest_stores_no_raw_pii_and_round_trips(
    pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
    owner: asyncpg.Connection,
) -> None:
    res = await _ingestor(pools, keyring, analyzer, settings).ingest(INGESTER, _doc())
    assert {"PERSON", "US_SSN", "EMAIL_ADDRESS", "PHONE_NUMBER"} <= res.pii_counts.keys()

    stored = await _all_stored_text(owner)
    for v in [*PII["names"], PII["ssn"], PII["email"], PII["phone"], "Okafor", "Whitfield"]:
        assert v not in stored, f"raw PII stored: {v!r}"

    # Chunk roles: HR section -> [hr] only; public section inherits [hr, intern].
    rows = await owner.fetch(
        "SELECT allowed_roles, text_redacted, pii_types FROM core.chunks WHERE document_id = $1"
        " ORDER BY ord", res.document_id,
    )
    assert [sorted(r["allowed_roles"]) for r in rows] == [["hr"], ["hr", "intern"]]
    assert "US_SSN" in rows[0]["pii_types"] and "US_SSN" not in rows[1]["pii_types"]

    tokens = {t for r in rows for t in find_tokens(r["text_redacted"])}
    async with scoped(pools.vault, "acme", acquire_timeout_s=1) as c:
        revealed = await vault.load(c, "acme", tokens, await keyring.enc_keys(c, "acme"))
    assert {r.value for r in revealed.values()} >= {*PII["names"], PII["ssn"], PII["email"]}

    audit = await owner.fetchrow(
        "SELECT * FROM audit.events WHERE action = 'ingest' AND $1 = ANY(document_ids)",
        res.document_id,
    )
    assert audit is not None and "US_SSN" in audit["pii_types_present"]


async def test_same_name_same_token_in_tenant_different_across(
    pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
    owner: asyncpg.Connection,
) -> None:
    ing = _ingestor(pools, keyring, analyzer, settings)
    a1 = await ing.ingest(INGESTER, _doc("Memo one"))
    a2 = await ing.ingest(INGESTER, _doc("Memo two"))
    g = await ing.ingest(GLOBEX_INGESTER, DocumentIn("Globex memo", ("hr",), (SectionIn(HR_TEXT),)))

    async def toks(doc: uuid.UUID) -> set[str]:
        rows = await owner.fetch("SELECT text_redacted FROM core.chunks WHERE document_id=$1", doc)
        return {t for r in rows for t in find_tokens(r["text_redacted"])}

    t1, t2, tg = await toks(a1.document_id), await toks(a2.document_id), await toks(g.document_id)
    assert t1 and t1 == t2
    assert not (t1 & tg)
    # Dedupe: the second acme doc added no new vault rows.
    assert a2.new_tokens == 0


async def test_delete_erases_only_unshared_tokens(
    pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
    owner: asyncpg.Connection,
) -> None:
    ing = _ingestor(pools, keyring, analyzer, settings)
    unique = await ing.ingest(
        INGESTER,
        DocumentIn("Note", ("hr",), (SectionIn("Priyanka Raghunathan joined the team."),)),
    )
    shared = await ing.ingest(INGESTER, DocumentIn("Note", ("hr",), (SectionIn(HR_TEXT),)))
    rows = await owner.fetch(
        "SELECT token FROM vault.token_refs r JOIN core.chunks c ON c.id = r.chunk_id"
        " WHERE c.document_id = $1", unique.document_id,
    )
    unique_tokens = [r["token"] for r in rows]
    assert unique_tokens
    erased = await ing.delete(INGESTER, unique.document_id)
    assert erased == len(unique_tokens)
    left = await owner.fetchval(
        "SELECT count(*) FROM vault.pii_tokens WHERE token = ANY($1)", unique_tokens
    )
    assert left == 0
    # Tokens of the other document are untouched.
    still = await owner.fetchval(
        "SELECT count(*) FROM vault.token_refs r JOIN core.chunks c ON c.id = r.chunk_id"
        " WHERE c.document_id = $1", shared.document_id,
    )
    assert still > 0


async def test_name_found_once_is_hidden_everywhere_in_doc(
    pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
    owner: asyncpg.Connection,
) -> None:
    # spaCy alone misses the name in the first and last sections.
    doc = DocumentIn("Onboarding", ("hr",), (
        SectionIn("Contact Priyanka Raghunathan today."),
        SectionIn("Priyanka Raghunathan joined the team."),
        SectionIn("Ask Raghunathan for the badge."),
    ))
    res = await _ingestor(pools, keyring, analyzer, settings).ingest(INGESTER, doc)
    rows = await owner.fetch(
        "SELECT text_redacted FROM core.chunks WHERE document_id = $1", res.document_id
    )
    text = " ".join(r["text_redacted"] for r in rows)
    assert "Priyanka" not in text and "Raghunathan" not in text


async def test_collision_fails_loudly(pools: Pools, keyring: TenantKeyring) -> None:
    tok = "PERSON_00000000deadbeef"
    async with scoped(pools.ingest, "acme", acquire_timeout_s=1) as c:
        enc = await keyring.enc_keys(c, "acme")
        await vault.store(c, "acme", [Found(tok, "PERSON", "Alice", b"\x01" * 32)], enc)
    with pytest.raises(IntegrityError, match="token_collision"):
        async with scoped(pools.ingest, "acme", acquire_timeout_s=1) as c:
            await vault.store(c, "acme", [Found(tok, "PERSON", "Bob", b"\x02" * 32)], enc)


async def test_master_key_rotation(
    pools: Pools, owner: asyncpg.Connection, settings: Settings, keyring: TenantKeyring,
) -> None:
    old = settings.master_keys.get_secret_value()
    new = Fernet.generate_key().decode()
    async with scoped(pools.vault, "acme", acquire_timeout_s=1) as c:
        before = await c.fetch("SELECT token, ciphertext FROM vault.pii_tokens LIMIT 5")
        old_keys = await keyring.enc_keys(c, "acme")
    assert before
    assert await rotate_master(owner, MasterKey(f"{new},{old}")) > 0
    # The old master key is gone from config. Only the new one can unwrap now.
    new_only = TenantKeyring(MasterKey(new))
    async with scoped(pools.vault, "acme", acquire_timeout_s=1) as c:
        keys = await new_only.enc_keys(c, "acme")
        got = await vault.load(c, "acme", [r["token"] for r in before], keys)
    assert len(got) == len(before)
    with pytest.raises(IntegrityError):
        async with scoped(pools.vault, "acme", acquire_timeout_s=1) as c:
            await TenantKeyring(MasterKey(old)).enc_keys(c, "acme")
    # Put things back so later tests (and the shared keyring) keep working.
    await rotate_master(owner, MasterKey(f"{old},{new}"))
    assert isinstance(old_keys, EncKeys)


async def test_shred_makes_pii_unrecoverable(
    pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
    owner: asyncpg.Connection, master: MasterKey,
) -> None:
    ing = _ingestor(pools, keyring, analyzer, settings)
    await ing.ingest(GLOBEX_INGESTER, DocumentIn("Keep", ("hr",), (SectionIn(HR_TEXT),)))
    globex_before = await owner.fetchval(
        "SELECT count(*) FROM vault.pii_tokens WHERE tenant_id = 'globex'"
    )
    with pytest.raises(Forbidden):
        await ing.shred_tenant_pii(GLOBEX_INGESTER, master)
    assert await ing.shred_tenant_pii(INGESTER, master) > 0
    assert await owner.fetchval(
        "SELECT count(*) FROM vault.pii_tokens WHERE tenant_id = 'acme'") == 0
    assert await owner.fetchval(
        "SELECT count(*) FROM vault.pii_tokens WHERE tenant_id = 'globex'") == globex_before
    # Chunks still exist, with placeholders, and new ingests work with the fresh key.
    assert await owner.fetchval("SELECT count(*) FROM core.chunks WHERE tenant_id='acme'") > 0
    res = await ing.ingest(INGESTER, _doc("After shred"))
    assert res.new_tokens > 0
    assert await owner.fetchval(
        "SELECT min(key_version) FROM vault.tenant_enc_keys WHERE tenant_id = 'acme'") == 2


async def test_validation_and_permissions(
    pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
) -> None:
    ing = _ingestor(pools, keyring, analyzer, settings)
    with pytest.raises(Forbidden):
        await ing.ingest(Principal("acme", "u", frozenset({"hr"})), _doc())
    bad = [
        DocumentIn("t", ("hr",), (SectionIn("x", ("finance",)),)),  # disjoint roles
        DocumentIn("t", (), (SectionIn("x"),)),
        DocumentIn("t", ("HR!",), (SectionIn("x"),)),
        DocumentIn("", ("hr",), (SectionIn("x"),)),
        DocumentIn("t", ("hr",), ()),
        DocumentIn("t", ("hr",), (SectionIn("x" * (settings.max_section_chars + 1)),)),
    ]
    for d in bad:
        with pytest.raises(BadRequest):
            await ing.ingest(INGESTER, d)


async def test_ingest_sheds_load_when_busy(
    pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
) -> None:
    ing = _ingestor(pools, keyring, analyzer, settings)
    tasks = [asyncio.create_task(ing.ingest(INGESTER, _doc(f"burst {i}"))) for i in range(6)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    shed = [r for r in results if isinstance(r, Overloaded)]
    ok = [r for r in results if not isinstance(r, BaseException)]
    assert shed and ok and len(shed) + len(ok) == 6


async def test_event_loop_stays_responsive_during_big_ingest(
    pools: Pools, keyring: TenantKeyring, analyzer: PiiAnalyzer, settings: Settings,
) -> None:
    """Presidio, chunking and embeddings must all stay off the loop."""
    embedder = ProcessEmbedder(settings.embed_model, dim=settings.embed_dim)
    await embedder.start()
    ing = _ingestor(pools, keyring, analyzer, settings, embedder)
    sections = tuple(SectionIn(f"Section {i}. {HR_TEXT} {PUBLIC_TEXT} " * 3) for i in range(40))

    async def worst_lag(work: object) -> float:
        worst = 0.0
        done = asyncio.Event()

        async def ticker() -> None:
            nonlocal worst
            while not done.is_set():
                t = time.perf_counter()
                await asyncio.sleep(0.005)
                worst = max(worst, time.perf_counter() - t - 0.005)

        tick = asyncio.create_task(ticker())
        try:
            await work  # type: ignore[misc]
        finally:
            done.set()
            await tick
        return worst

    try:
        # Baseline: how late a 5 ms sleep wakes up on this machine right now, with
        # nothing of ours running. A busy laptop adds its own lag; we only judge ours.
        base = await worst_lag(asyncio.sleep(2.0))
        lag = await worst_lag(ing.ingest(INGESTER, DocumentIn("Big", ("hr",), sections)))
    finally:
        embedder.close()
    msg = f"ingest blocked the loop {lag * 1000:.0f} ms (base {base * 1000:.0f} ms)"
    assert lag - base < 0.05, msg
