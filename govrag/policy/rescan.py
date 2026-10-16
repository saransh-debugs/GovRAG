"""Background re-scan: when a tenant adds an `extra_redact` type, hide it in old chunks too.

How it stays correct under crashes and concurrency:
  - Work is found by `chunks.redaction_rev < target`, not by a cursor. A crashed job
    resumes exactly where the data says it is, and documents ingested mid-job (under a
    stale cached policy) are picked up too.
  - Each document is one transaction, and the UPDATE re-checks redaction_rev, so two
    workers racing on one document can't apply it twice.
  - A job applies the tenant's *current* full extra_redact list at the current policy
    version. If two edits each add a type, whichever job runs covers both.
  - Leases: a claimed job holds a 60 s lease, renewed per document. A dead worker's job
    gets re-claimed after the lease runs out.
  - Rate limited (pause between documents), so a big tenant's re-scan can't starve
    live ingests of Presidio workers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

import asyncpg

from govrag import audit
from govrag.db import Pools, scoped
from govrag.embed import Embedder
from govrag.errors import GovRagError
from govrag.ingest_cpu import rescan_redact
from govrag.keys import TenantKeyring
from govrag.pii import vault
from govrag.pii.analyzer import PiiAnalyzer
from govrag.pii.tokens import entity_type_of, find_tokens
from govrag.policy.store import read_active

log = logging.getLogger(__name__)
LEASE = "60 seconds"


@dataclass(frozen=True, slots=True)
class Job:
    tenant_id: str
    id: uuid.UUID


class Rescanner:
    def __init__(
        self,
        *,
        pools: Pools,
        keyring: TenantKeyring,
        analyzer: PiiAnalyzer,
        embedder: Embedder,
        pause_s: float = 0.05,
        poll_s: float = 2.0,
    ) -> None:
        self._pools = pools
        self._keyring = keyring
        self._analyzer = analyzer
        self._embedder = embedder
        self._pause = pause_s
        self._poll = poll_s
        self._task: asyncio.Task[None] | None = None

    def _tx(self, tenant: str) -> AbstractAsyncContextManager[asyncpg.Connection]:
        return scoped(self._pools.ingest, tenant, acquire_timeout_s=self._pools.acquire_timeout_s)

    async def claim(self) -> Job | None:
        async with self._pools.ingest.acquire(timeout=self._pools.acquire_timeout_s) as c:
            row = await c.fetchrow(
                f"""
                UPDATE core.rescan_jobs SET status = 'running',
                       lease_until = now() + interval '{LEASE}', updated_at = now()
                 WHERE (tenant_id, id) = (
                       SELECT tenant_id, id FROM core.rescan_jobs
                        WHERE status = 'pending'
                           OR (status = 'running' AND lease_until < now())
                        ORDER BY created_at
                        LIMIT 1
                          FOR UPDATE SKIP LOCKED)
                RETURNING tenant_id, id
                """  # noqa: S608  (LEASE is a constant)
            )
        return Job(row["tenant_id"], row["id"]) if row else None

    async def run_job(self, job: Job, *, max_docs: int | None = None) -> bool:
        """Process documents until none are behind. Returns True when the job is done.

        max_docs: stop early (tests use it to simulate a crash halfway).
        """
        t = job.tenant_id
        async with self._tx(t) as c:
            active = await read_active(c, t)
            hmac_key = await self._keyring.hmac_key(c, t)
            enc = await self._keyring.enc_keys(c, t)
        target, types = active.version, active.policy.extra_redact
        done_docs = 0
        while types:
            if max_docs is not None and done_docs >= max_docs:
                return False  # stopped early; the job stays 'running' until its lease ends
            async with self._tx(t) as c:
                doc = await c.fetchval(
                    "SELECT document_id FROM core.chunks"
                    " WHERE tenant_id = $1 AND redaction_rev < $2 LIMIT 1",
                    t, target,
                )
                chunks = [] if doc is None else await c.fetch(
                    "SELECT id, text_redacted FROM core.chunks WHERE tenant_id = $1"
                    " AND document_id = $2 AND redaction_rev < $3 ORDER BY ord",
                    t, doc, target,
                )
            if doc is None:
                break
            texts = [r["text_redacted"] for r in chunks]
            spans = await self._analyzer.analyze(texts, types, lane="ingest")
            new_texts, found = await self._analyzer.run(
                "ingest", rescan_redact, texts, spans, hmac_key, timeout_s=120.0
            )
            changed = [i for i, (a, b) in enumerate(zip(texts, new_texts, strict=True))
                       if a != b]
            vectors = await self._embedder.embed_passages([new_texts[i] for i in changed])
            async with self._tx(t) as c:
                await vault.store(c, t, found.values(), enc)
                refs: list[tuple[str, uuid.UUID]] = []
                for i, vec in zip(changed, vectors, strict=True):
                    toks = set(find_tokens(new_texts[i])) & found.keys()
                    refs.extend((tok, chunks[i]["id"]) for tok in toks)
                    await c.execute(
                        "UPDATE core.chunks SET text_redacted = $3, embedding = $4,"
                        " pii_types = ARRAY(SELECT DISTINCT unnest(pii_types || $5::text[])),"
                        " redaction_rev = $6"
                        " WHERE tenant_id = $1 AND id = $2 AND redaction_rev < $6",
                        t, chunks[i]["id"], new_texts[i], vec,
                        sorted({entity_type_of(x) for x in toks}), target,
                    )
                await vault.add_refs(c, t, refs)
                # Unchanged chunks: just record that they've been checked at this version.
                await c.execute(
                    "UPDATE core.chunks SET redaction_rev = $3"
                    " WHERE tenant_id = $1 AND id = ANY($2) AND redaction_rev < $3",
                    t, [r["id"] for r in chunks], target,
                )
                await c.execute(
                    f"UPDATE core.rescan_jobs SET cursor_doc = $3, docs_done = docs_done + 1,"  # noqa: S608
                    f" chunks_done = chunks_done + $4, lease_until = now() + interval '{LEASE}',"
                    " updated_at = now() WHERE tenant_id = $1 AND id = $2",
                    t, job.id, doc, len(changed),
                )
            done_docs += 1
            await asyncio.sleep(self._pause)
        async with self._tx(t) as c:
            await c.execute(
                "UPDATE core.rescan_jobs SET status = 'done', lease_until = NULL,"
                " updated_at = now() WHERE tenant_id = $1 AND id = $2",
                t, job.id,
            )
            await audit.record(c, audit.AuditEvent(
                tenant_id=t, request_id=job.id, user_id="system:rescan", action="rescan",
                pii_types_present=frozenset(types), policy_version=target,
            ))
        return True

    async def run_forever(self) -> None:
        while True:
            try:
                job = await self.claim()
                if job is None:
                    await asyncio.sleep(self._poll)
                    continue
                await self.run_job(job)
            except asyncio.CancelledError:
                raise
            except (GovRagError, asyncpg.PostgresError, OSError) as e:
                # The lease runs out and the job gets retried. Never crash the app for it.
                log.warning("rescan failed: %s", type(e).__name__)
                await asyncio.sleep(self._poll)

    def start(self) -> None:
        self._task = asyncio.create_task(self.run_forever(), name="rescan")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
