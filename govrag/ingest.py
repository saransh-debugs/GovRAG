"""Ingest: redact -> chunk -> embed -> one transaction.

Order matters:
  1. Short transaction: read the tenant's keys. Then release the connection.
  2. No DB connection held: Presidio (seconds), chunking, embedding. Holding a transaction
     open across seconds of CPU would pin a pooled connection and trip
     idle_in_transaction_session_timeout under load.
  3. One transaction: document + chunks + vault rows + token refs + audit row. Either all
     of it exists or none of it does. No chunk ever references a token the vault lacks.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from dataclasses import dataclass

import asyncpg
from cryptography.fernet import Fernet

from govrag import audit
from govrag.auth import Principal
from govrag.chunking import Chunker
from govrag.config import Settings
from govrag.db import IDENT, Pools, scoped
from govrag.embed import Embedder
from govrag.errors import BadRequest, Forbidden, NotFound, Overloaded, Unavailable
from govrag.ingest_cpu import prepare_document
from govrag.keys import MasterKey, TenantKeyring
from govrag.pii import vault
from govrag.pii.analyzer import PiiAnalyzer
from govrag.pii.entities import DEFAULT_REDACT
from govrag.pii.tokens import entity_type_of, find_tokens, neutralize
from govrag.tracing import annotate, step, tracer

INGEST_ROLE = "ingest"
# Crypto-shredding is irreversible, so it needs its own role, not just "ingest".
SHRED_ROLE = "dpo"  # data protection officer
MAX_TITLE_CHARS = 500
MAX_SECTIONS = 1000
MAX_ROLES = 32
_FK_RETRIES = 3


@dataclass(frozen=True, slots=True)
class SectionIn:
    text: str
    allowed_roles: tuple[str, ...] | None = None  # None = same as the document


@dataclass(frozen=True, slots=True)
class DocumentIn:
    title: str
    allowed_roles: tuple[str, ...]
    sections: tuple[SectionIn, ...]


@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: uuid.UUID
    chunk_ids: tuple[uuid.UUID, ...]
    pii_counts: dict[str, int]
    new_tokens: int


def _check_roles(roles: tuple[str, ...]) -> None:
    if not roles or len(roles) > MAX_ROLES or not all(IDENT.match(r) for r in roles):
        raise BadRequest("bad_roles")


def validate(doc: DocumentIn, s: Settings) -> list[tuple[str, ...]]:
    """Returns each section's effective roles (section roles ∩ document roles)."""
    if not doc.title.strip() or len(doc.title) > MAX_TITLE_CHARS:
        raise BadRequest("bad_title")
    if not doc.sections or len(doc.sections) > MAX_SECTIONS:
        raise BadRequest("bad_sections")
    _check_roles(doc.allowed_roles)
    total = 0
    effective: list[tuple[str, ...]] = []
    for sec in doc.sections:
        if len(sec.text) > s.max_section_chars:
            raise BadRequest("section_too_large")
        total += len(sec.text)
        if sec.allowed_roles is None:
            effective.append(tuple(sorted(set(doc.allowed_roles))))
            continue
        _check_roles(sec.allowed_roles)
        eff = tuple(sorted(set(sec.allowed_roles) & set(doc.allowed_roles)))
        # A section nobody may read is a caller bug, not something to store silently.
        if not eff:
            raise BadRequest("section_roles_disjoint")
        effective.append(eff)
    if total > s.max_document_chars:
        raise BadRequest("document_too_large")
    return effective


class Ingestor:
    def __init__(
        self,
        *,
        pools: Pools,
        keyring: TenantKeyring,
        analyzer: PiiAnalyzer,
        embedder: Embedder,
        chunker: Chunker,
        settings: Settings,
    ) -> None:
        self._pools = pools
        self._keyring = keyring
        self._analyzer = analyzer
        self._embedder = embedder
        self._chunker = chunker
        self._s = settings
        # Ingest is heavy (Presidio + embeddings). Past this many at once, 503 instead of
        # queueing whole documents in memory.
        self._slots = asyncio.Semaphore(settings.max_inflight_ingests)

    async def ingest(
        self,
        p: Principal,
        doc: DocumentIn,
        *,
        extra_redact: frozenset[str] = frozenset(),
        redaction_rev: int = 0,
        request_id: uuid.UUID | None = None,
    ) -> IngestResult:
        """redaction_rev: the policy version whose extra_redact list was applied. The
        re-scan job skips chunks already at (or past) the version it's applying."""
        if not p.has(INGEST_ROLE):
            raise Forbidden("ingest_role_required")
        section_roles = validate(doc, self._s)
        if self._slots.locked():
            raise Overloaded("ingest_busy", retry_after_s=5.0)
        async with self._slots:
            with tracer.start_as_current_span("ingest", attributes={
                "govrag.tenant_id": p.tenant_id, "govrag.user_id": p.user_id,
            }) as sp:
                res = await self._ingest(p, doc, section_roles, extra_redact, redaction_rev,
                                         request_id)
                annotate(sp, document_ids=[res.document_id], chunks=len(res.chunk_ids),
                         pii_types=set(res.pii_counts), new_tokens=res.new_tokens)
                return res

    async def _ingest(
        self,
        p: Principal,
        doc: DocumentIn,
        section_roles: list[tuple[str, ...]],
        extra_redact: frozenset[str],
        redaction_rev: int,
        request_id: uuid.UUID | None,
    ) -> IngestResult:
        tenant = p.tenant_id
        async with scoped(
            self._pools.ingest, tenant, acquire_timeout_s=self._pools.acquire_timeout_s
        ) as c:
            hmac_key = await self._keyring.hmac_key(c, tenant)
            enc = await self._keyring.enc_keys(c, tenant)

        # Break up token-shaped strings in the input, so a document can't carry a forged
        # "[US_SSN_<someone else's token>]". The vault refs check (load_for_chunks) is the
        # real guard; this is the second check.
        texts = [neutralize(doc.title), *(neutralize(s.text) for s in doc.sections)]
        with step("pii.analyze", lane="ingest"):
            spans = await self._analyzer.analyze(texts, DEFAULT_REDACT | extra_redact,
                                                 lane="ingest")
        # Propagate + redact + chunk in a worker process, not a thread: see ingest_cpu.py.
        with step("redact_and_chunk"):
            prep = await self._analyzer.run(
                "ingest", prepare_document, texts, spans, hmac_key,
                self._chunker.chunk_size, self._chunker.chunk_overlap, timeout_s=120.0,
            )
        found = prep.found
        pieces = prep.chunks
        rows: list[tuple[uuid.UUID, int, str, tuple[str, ...]]] = []
        for sec_i, chunks in enumerate(pieces):
            for text in chunks:
                rows.append((uuid.uuid4(), len(rows), text, section_roles[sec_i]))
        if not rows:
            raise BadRequest("empty_document")
        with step("embed", chunks=len(rows)):
            vectors = await self._embedder.embed_passages([r[2] for r in rows])

        refs: list[tuple[str, uuid.UUID]] = []
        chunk_types: list[list[str]] = []
        for cid, _, text, _ in rows:
            toks = set(find_tokens(text))
            # Only tokens we minted count. Text that merely looks like a token (e.g. the
            # document itself contains "[PERSON_0123456789abcdef]") gets no vault ref.
            toks &= found.keys()
            refs.extend((t, cid) for t in toks)
            chunk_types.append(sorted({entity_type_of(t) for t in toks}))

        doc_id = uuid.uuid4()
        rid = request_id or uuid.uuid4()
        new_tokens = 0
        for attempt in range(_FK_RETRIES):
            try:
                async with scoped(
                    self._pools.ingest, tenant, acquire_timeout_s=self._pools.acquire_timeout_s
                ) as c:
                    await c.execute(
                        "INSERT INTO core.documents (tenant_id, id, title, allowed_roles,"
                        " created_by) VALUES ($1, $2, $3, $4, $5)",
                        tenant, doc_id, prep.title, sorted(set(doc.allowed_roles)), p.user_id,
                    )
                    # COPY would be faster, but Postgres refuses COPY FROM into a table with
                    # RLS for a non-BYPASSRLS login. executemany pipelines the inserts in
                    # one round trip, which is the next best thing.
                    await c.executemany(
                        "INSERT INTO core.chunks (tenant_id, id, document_id, ord, text_redacted,"
                        " embedding, allowed_roles, pii_types, redaction_rev)"
                        " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
                        [
                            (tenant, cid, doc_id, ord_, text, vec, list(roles), types,
                             redaction_rev)
                            for (cid, ord_, text, roles), vec, types in zip(
                                rows, vectors, chunk_types, strict=True
                            )
                        ],
                    )
                    new_tokens = await vault.store(c, tenant, found.values(), enc)
                    await vault.add_refs(c, tenant, refs)
                    await audit.record(
                        c,
                        audit.AuditEvent(
                            tenant_id=tenant,
                            request_id=rid,
                            user_id=p.user_id,
                            action="ingest",
                            document_ids=(doc_id,),
                            chunk_ids=tuple(r[0] for r in rows),
                            pii_types_present=frozenset(f.entity_type for f in found.values()),
                        ),
                    )
                break
            except asyncpg.ForeignKeyViolationError:
                # A concurrent delete erased a token between our ON CONFLICT DO NOTHING and
                # our ref insert. Optimistic retry: the next attempt re-inserts the token.
                if attempt == _FK_RETRIES - 1:
                    raise Unavailable("ingest_conflict") from None
        counts = Counter(f.entity_type for f in found.values())
        return IngestResult(doc_id, tuple(r[0] for r in rows), dict(counts), new_tokens)

    async def delete(
        self, p: Principal, document_id: uuid.UUID, *, request_id: uuid.UUID | None = None
    ) -> int:
        if not p.has(INGEST_ROLE):
            raise Forbidden("ingest_role_required")
        async with scoped(
            self._pools.ingest, p.tenant_id, acquire_timeout_s=self._pools.acquire_timeout_s
        ) as c:
            exists = await c.fetchval(
                "SELECT 1 FROM core.documents WHERE tenant_id = $1 AND id = $2",
                p.tenant_id, document_id,
            )
            if not exists:
                raise NotFound("document_not_found")
            erased = await vault.delete_document(c, p.tenant_id, document_id)
            await audit.record(
                c,
                audit.AuditEvent(
                    tenant_id=p.tenant_id,
                    request_id=request_id or uuid.uuid4(),
                    user_id=p.user_id,
                    action="delete",
                    document_ids=(document_id,),
                ),
            )
        return erased

    async def shred_tenant_pii(
        self, p: Principal, master: MasterKey, *, request_id: uuid.UUID | None = None
    ) -> int:
        """Make every stored PII value of this tenant unrecoverable. Returns rows erased."""
        if not p.has(SHRED_ROLE):
            raise Forbidden("dpo_role_required")
        async with scoped(
            self._pools.ingest, p.tenant_id, acquire_timeout_s=self._pools.acquire_timeout_s
        ) as c:
            n = await vault.shred_tenant(c, p.tenant_id, master.wrap(Fernet.generate_key()))
            await audit.record(
                c,
                audit.AuditEvent(
                    tenant_id=p.tenant_id,
                    request_id=request_id or uuid.uuid4(),
                    user_id=p.user_id,
                    action="shred",
                ),
            )
        self._keyring.invalidate(p.tenant_id)
        return n
