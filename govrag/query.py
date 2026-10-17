"""The query path.

  admit -> policy -> neutralize + redact question -> embed -> search (RLS)
        -> audit "retrieve"                       (before the LLM vendor sees anything)
        -> LLM first token  ||  vault decrypt     (in parallel: hides vault latency)
        -> audit "rehydrate"                      (before the user sees anything)
        -> stream: rehydrate each piece -> client

Every answered request writes exactly two audit rows, both before data moves: one before
redacted text leaves for the LLM vendor, one before any decrypted PII leaves for the user.
Either write failing ends the request with 503.

Timeouts: a single deadline per request, applied to each await separately. Wrapping the
whole generator in asyncio.timeout() is a trap: across a `yield`, the cancel can land in
the *consumer's* code (e.g. mid socket write) instead of here.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace
from typing import Literal

import asyncpg
from llama_index.core.schema import QueryBundle
from opentelemetry.trace import Span, StatusCode

from govrag import audit, prompt
from govrag.auth import Principal
from govrag.config import Settings
from govrag.db import IDENT, Pools, scoped
from govrag.embed import Embedder
from govrag.errors import BadRequest, Forbidden, GovRagError, Unavailable
from govrag.keys import TenantKeyring
from govrag.limits import Admission
from govrag.llm import LlmRouter
from govrag.pii import vault
from govrag.pii.analyzer import PiiAnalyzer
from govrag.pii.entities import DEFAULT_REDACT
from govrag.pii.redact import RedactedQuestion, redact_question
from govrag.pii.tokens import entity_type_of, neutralize
from govrag.policy.cache import PolicyCache
from govrag.policy.model import ActivePolicy, Policy
from govrag.policy.store import ADMIN_ROLE
from govrag.rehydrate import Rehydrator
from govrag.retrieval import GovRetriever, Hit, Retriever, hits_from_nodes
from govrag.tracing import annotate, step, tracer

Rehydration = Literal["ok", "unavailable", "none"]


@dataclass(frozen=True, slots=True)
class Citation:
    n: int
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    title: str


@dataclass(slots=True)
class AnswerMeta:
    request_id: uuid.UUID
    model: str | None
    policy_version: int
    rehydration: Rehydration
    citations: list[Citation] = field(default_factory=list)
    unmasked_types: list[str] = field(default_factory=list)
    masked_types: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Delta:
    text: str


@dataclass(frozen=True, slots=True)
class Done:
    meta: AnswerMeta


Event = Delta | Done


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    meta: AnswerMeta


@dataclass(frozen=True, slots=True)
class DryRun:
    roles: list[str]
    document_ids: list[uuid.UUID]
    chunk_ids: list[uuid.UUID]
    pii_types_present: list[str]
    pii_types_unmasked: list[str]


class Deadline:
    def __init__(self, seconds: float) -> None:
        self._end = time.monotonic() + seconds

    def left(self) -> float:
        rem = self._end - time.monotonic()
        if rem <= 0:
            raise Unavailable("deadline_exceeded")
        return rem


@dataclass(slots=True)
class _Prepared:
    active: ActivePolicy
    may_unmask: frozenset[str]
    question: RedactedQuestion
    hits: list[Hit]


class QueryService:
    def __init__(
        self,
        *,
        pools: Pools,
        keyring: TenantKeyring,
        analyzer: PiiAnalyzer,
        embedder: Embedder,
        retriever: Retriever,
        router: LlmRouter,
        policies: PolicyCache,
        admission: Admission,
        settings: Settings,
    ) -> None:
        self._pools = pools
        self._keyring = keyring
        self._analyzer = analyzer
        self._embedder = embedder
        self._retriever = retriever
        self._router = router
        self._policies = policies
        self._admission = admission
        self._s = settings

    # --- shared front half -----------------------------------------------------------

    async def _hmac(self, tenant: str) -> bytes:
        key = self._keyring.cached_hmac(tenant)
        if key is not None:
            return key
        async with scoped(self._pools.app, tenant,
                          acquire_timeout_s=self._pools.acquire_timeout_s) as c:
            return await self._keyring.hmac_key(c, tenant)

    def _check_question(self, question: str, top_k: int | None) -> int:
        if not question.strip() or len(question) > self._s.max_question_chars:
            raise BadRequest("bad_question")
        k = top_k or self._s.top_k_default
        if not 1 <= k <= self._s.top_k_max:
            raise BadRequest("bad_top_k")
        return k

    async def _prepare(
        self, p: Principal, question: str, k: int, active: ActivePolicy, dl: Deadline,
        root: Span,
    ) -> _Prepared:
        may = active.policy.may_unmask(p.roles)
        # Neutralize first: it changes the text, so it must run before span offsets exist.
        q = neutralize(question)
        with step("redact_question", root, lane="query") as sp:
            spans = await self._analyzer.analyze(
                [q], DEFAULT_REDACT | active.policy.extra_redact, lane="query",
                timeout_s=min(dl.left(), 5.0),
            )
            rq = redact_question(q, spans[0], await self._hmac(p.tenant_id), may)
            annotate(sp, pii_types=rq.types)
        with step("embed", root):
            qvec = await asyncio.wait_for(self._embedder.embed_query(rq.text), dl.left())
        with step("retrieve", root, k=k) as sp:
            nodes = await asyncio.wait_for(
                GovRetriever(self._retriever, p, k).aretrieve(
                    QueryBundle(query_str=rq.text, embedding=qvec.tolist())
                ),
                dl.left(),
            )
            hits = hits_from_nodes(nodes)
            annotate(sp, hits=len(hits), document_ids={h.document_id for h in hits},
                     chunk_ids=[h.chunk_id for h in hits])
        return _Prepared(active, may, rq, hits)

    async def _audit(self, p: Principal, ev: audit.AuditEvent, root: Span | None = None) -> None:
        try:
            with step(f"audit.{ev.action}", root):
                async with scoped(self._pools.app, p.tenant_id, p.roles,
                                  acquire_timeout_s=self._pools.acquire_timeout_s) as c:
                    await audit.record(c, ev)
        except GovRagError as e:
            raise Unavailable("audit_unavailable") from e
        except (asyncpg.PostgresError, OSError) as e:
            raise Unavailable("audit_unavailable") from e

    async def _reveal(
        self, p: Principal, hits: list[Hit], may: frozenset[str]
    ) -> tuple[dict[str, str], Rehydration]:
        # Data minimization: only decrypt types that both appear in these chunks and
        # that this user may see. An intern's request never decrypts an SSN.
        present = frozenset().union(*(h.pii_types for h in hits)) if hits else frozenset()
        types = may & present
        if not types:
            return {}, "none"
        try:
            async with scoped(self._pools.vault, p.tenant_id,
                              acquire_timeout_s=self._pools.acquire_timeout_s) as c:
                keys = await self._keyring.enc_keys(c, p.tenant_id)
                got = await vault.load_for_chunks(
                    c, p.tenant_id, [h.chunk_id for h in hits], types, keys
                )
        except (GovRagError, asyncpg.PostgresError, OSError):
            # Degrade, don't fail: the answer still works with placeholders. This is the
            # one failure where "less data" is the safe direction.
            return {}, "unavailable"
        return {t: r.value for t, r in got.items()}, "ok"

    # --- answer ----------------------------------------------------------------------

    async def stream(
        self, p: Principal, question: str, *, top_k: int | None = None,
        request_id: uuid.UUID | None = None,
    ) -> AsyncGenerator[Event, None]:
        rid = request_id or uuid.uuid4()
        k = self._check_question(question, top_k)
        # Root span is started, never made current: it lives across yields (see tracing.py).
        root = tracer.start_span("query", attributes={
            "govrag.tenant_id": p.tenant_id, "govrag.user_id": p.user_id,
            "govrag.request_id": str(rid),
        })
        try:
            async for ev in self._stream(p, question, k, rid, root):
                yield ev
            annotate(root, outcome="ok")
        except GovRagError as e:
            annotate(root, outcome="error", error_code=e.code)
            root.set_status(StatusCode.ERROR)
            raise
        except BaseException:
            annotate(root, outcome="error", error_code="internal")
            root.set_status(StatusCode.ERROR)
            raise
        finally:
            root.end()

    async def _stream(
        self, p: Principal, question: str, k: int, rid: uuid.UUID, root: Span
    ) -> AsyncGenerator[Event, None]:
        dl = Deadline(self._s.request_deadline_s)
        with self._admission.admit(p.tenant_id):
            with step("policy", root) as sp:
                active = await self._policies.get(p.tenant_id)
                annotate(sp, policy_version=active.version)
            annotate(root, policy_version=active.version)
            prep = await self._prepare(p, question, k, active, dl, root)
            hits = prep.hits
            chunks = [prompt.ContextChunk(i + 1, h.title, h.text) for i, h in enumerate(hits)]
            user, used = prompt.build(prep.question.text, chunks,
                                      max_chars=self._s.max_context_chars)
            hits = hits[:used]
            meta = AnswerMeta(
                request_id=rid, model=None, policy_version=active.version, rehydration="none",
                citations=[Citation(i + 1, h.document_id, h.chunk_id, h.title)
                           for i, h in enumerate(hits)],
            )
            await self._audit(p, audit.AuditEvent(
                tenant_id=p.tenant_id, request_id=rid, user_id=p.user_id, action="retrieve",
                document_ids=tuple(h.document_id for h in hits),
                chunk_ids=tuple(h.chunk_id for h in hits),
                pii_types_present=frozenset().union(*(h.pii_types for h in hits))
                if hits else frozenset(),
                policy_version=active.version,
            ), root)
            if not hits:
                yield Delta(prompt.NO_CONTEXT_ANSWER)
                yield Done(meta)
                return

            reveal_task = asyncio.create_task(self._reveal(p, hits, prep.may_unmask))
            try:
                with step("llm.first_token", root) as sp:
                    llm = await self._router.open(
                        active.policy.llm.models, active.policy.llm.region,
                        system=prompt.SYSTEM, user=user, max_tokens=self._s.llm_max_tokens,
                        first_token_timeout_s=dl.left(),
                    )
                    annotate(sp, model=llm.model)
            except BaseException:
                reveal_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, GovRagError):
                    await reveal_task
                raise
            try:
                with step("vault.wait", root):
                    reveal, status = await reveal_task
                # The asker's own typed values come last, so a vault value wins if both exist.
                rh = Rehydrator({**prep.question.own, **reveal})
                meta.model, meta.rehydration = llm.model, status
                await self._audit(p, audit.AuditEvent(
                    tenant_id=p.tenant_id, request_id=rid, user_id=p.user_id,
                    action="rehydrate", chunk_ids=tuple(h.chunk_id for h in hits),
                    pii_types_unmasked=frozenset(entity_type_of(t) for t in reveal),
                    policy_version=active.version, model=llm.model,
                ), root)
                async for piece in llm:
                    dl.left()
                    out = rh.feed(piece)
                    if out:
                        yield Delta(out)
                tail = rh.flush()
                if tail:
                    yield Delta(tail)
            finally:
                await llm.aclose()
            # Titles are stored redacted too. Same allowlist and aliases as the answer, so
            # a citation can't show a raw token ID either.
            meta.citations = [replace(c, title=rh.whole(c.title)) for c in meta.citations]
            meta.unmasked_types = sorted(rh.unmasked_types)
            meta.masked_types = sorted(rh.masked_types)
            annotate(root, model=meta.model, rehydration=meta.rehydration,
                     pii_types_unmasked=meta.unmasked_types, pii_types_masked=meta.masked_types,
                     chunk_ids=[c.chunk_id for c in meta.citations])
            yield Done(meta)

    async def answer(
        self, p: Principal, question: str, *, top_k: int | None = None,
        request_id: uuid.UUID | None = None,
    ) -> Answer:
        parts: list[str] = []
        meta: AnswerMeta | None = None
        gen = self.stream(p, question, top_k=top_k, request_id=request_id)
        try:
            async for ev in gen:
                if isinstance(ev, Delta):
                    parts.append(ev.text)
                else:
                    meta = ev.meta
        finally:
            await gen.aclose()
        assert meta is not None
        return Answer("".join(parts), meta)

    # --- policy dry run --------------------------------------------------------------

    async def dry_run(
        self, admin: Principal, policy: Policy, roles: list[str], question: str,
        *, top_k: int | None = None,
    ) -> DryRun:
        """What would `roles` retrieve under `policy`? IDs and PII *types* only.

        It never returns text or values and never calls the LLM. If it did, a
        policy_admin could "dry run" as hr and read SSNs they aren't allowed to see.
        """
        if not admin.has(ADMIN_ROLE):
            raise Forbidden("policy_admin_required")
        if not roles or len(roles) > 32 or not all(IDENT.match(r) for r in roles):
            raise BadRequest("bad_roles")
        k = self._check_question(question, top_k)
        dl = Deadline(self._s.request_deadline_s)
        who = Principal(admin.tenant_id, admin.user_id, frozenset(roles))
        with self._admission.admit(admin.tenant_id), tracer.start_as_current_span(
            "dry_run", attributes={"govrag.tenant_id": admin.tenant_id,
                                   "govrag.user_id": admin.user_id},
        ) as root:
            prep = await self._prepare(who, question, k, ActivePolicy(version=0, policy=policy),
                                       dl, root)
        present = frozenset().union(*(h.pii_types for h in prep.hits)) if prep.hits \
            else frozenset()
        result = DryRun(
            roles=sorted(roles),
            document_ids=list(dict.fromkeys(h.document_id for h in prep.hits)),
            chunk_ids=[h.chunk_id for h in prep.hits],
            pii_types_present=sorted(present),
            pii_types_unmasked=sorted(present & prep.may_unmask),
        )
        await self._audit(admin, audit.AuditEvent(
            tenant_id=admin.tenant_id, request_id=uuid.uuid4(), user_id=admin.user_id,
            action="dry_run", document_ids=tuple(result.document_ids),
            chunk_ids=tuple(result.chunk_ids),
            pii_types_present=present, pii_types_unmasked=frozenset(result.pii_types_unmasked),
        ))
        return result
