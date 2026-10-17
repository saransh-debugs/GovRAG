"""FastAPI app: service wiring, error mapping, request IDs, body size limits."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from govrag.auth import JwtVerifier, Principal
from govrag.chunking import Chunker
from govrag.config import Settings, get_settings
from govrag.db import Pools
from govrag.embed import Embedder, ProcessEmbedder
from govrag.errors import GovRagError, Unauthorized
from govrag.ingest import Ingestor
from govrag.keys import MasterKey, TenantKeyring
from govrag.limits import Admission
from govrag.llm import FakeLLM, LlmRouter, ModelFactory, vertex_factory
from govrag.pii.analyzer import PiiAnalyzer
from govrag.policy.cache import PolicyCache
from govrag.policy.rescan import Rescanner
from govrag.policy.store import PolicyStore
from govrag.query import QueryService
from govrag.retrieval import Retriever

log = logging.getLogger("govrag")


@dataclass
class Services:
    settings: Settings
    pools: Pools
    master: MasterKey
    keyring: TenantKeyring
    analyzer: PiiAnalyzer
    embedder: Embedder
    policies: PolicyCache
    store: PolicyStore
    ingestor: Ingestor
    query: QueryService
    verifier: JwtVerifier
    rescanner: Rescanner | None
    _closers: list[Callable[[], Awaitable[None] | None]]

    async def close(self) -> None:
        for c in reversed(self._closers):
            r = c()
            if r is not None:
                await r


async def build_services(
    s: Settings,
    *,
    verifier: JwtVerifier | None = None,
    llm_factory: ModelFactory | None = None,
    embedder: Embedder | None = None,
    analyzer: PiiAnalyzer | None = None,
    background: bool = True,
) -> Services:
    closers: list[Callable[[], Awaitable[None] | None]] = []
    pools = await Pools.create(s)
    closers.append(pools.close)
    master = MasterKey(s.master_keys.get_secret_value())
    keyring = TenantKeyring(master)
    if analyzer is None:
        analyzer = PiiAnalyzer(model=s.spacy_model, ingest_workers=s.pii_ingest_workers,
                               query_workers=s.pii_query_workers)
        await analyzer.start()
        closers.append(analyzer.close)
    if embedder is None:
        pe = ProcessEmbedder(s.embed_model, dim=s.embed_dim)
        await pe.start()
        closers.append(pe.close)
        embedder = pe
    if llm_factory is None:
        if s.llm_backend == "fake":
            fake = FakeLLM()
            llm_factory = lambda _name, _region: fake  # noqa: E731
        else:
            llm_factory = vertex_factory(
                project=s.vertex_project, timeout_s=s.llm_timeout_s,
                model_ids=s.vertex_model_ids, claude_regions=s.vertex_claude_regions,
                gemini_locations=s.vertex_gemini_locations,
            )
    router = LlmRouter(
        llm_factory, concurrency=s.llm_concurrency,
        first_token_timeout_s=s.llm_first_token_timeout_s,
        stall_timeout_s=s.llm_stall_timeout_s,
        breaker_failures=s.breaker_failures, breaker_cooldown_s=s.breaker_cooldown_s,
    )
    policies = PolicyCache(pools, s.dsn("govrag_app", s.db_app_password),
                           ttl_s=s.policy_cache_ttl_s)
    await policies.start()
    closers.append(policies.stop)
    rescanner = None
    if background:
        rescanner = Rescanner(pools=pools, keyring=keyring, analyzer=analyzer, embedder=embedder)
        rescanner.start()
        closers.append(rescanner.stop)
    return Services(
        settings=s, pools=pools, master=master, keyring=keyring, analyzer=analyzer,
        embedder=embedder, policies=policies, store=PolicyStore(pools),
        ingestor=Ingestor(pools=pools, keyring=keyring, analyzer=analyzer, embedder=embedder,
                          chunker=Chunker(), settings=s),
        query=QueryService(
            pools=pools, keyring=keyring, analyzer=analyzer, embedder=embedder,
            retriever=Retriever(pools, ef_search=s.hnsw_ef_search,
                                max_scan_tuples=s.hnsw_max_scan_tuples),
            router=router, policies=policies,
            admission=Admission(global_max=s.max_inflight_queries,
                                per_tenant_max=s.max_inflight_per_tenant),
            settings=s,
        ),
        verifier=verifier or JwtVerifier.from_file(
            s.jwt_public_key_path, issuer=s.jwt_issuer, audience=s.jwt_audience,
            leeway_s=s.jwt_leeway_s, max_lifetime_s=s.jwt_max_lifetime_s,
        ),
        rescanner=rescanner,
        _closers=closers,
    )


# --- middleware --------------------------------------------------------------------------

# Query bodies are tiny; documents are big. One global limit would either reject real
# documents or let a 2 MB "question" through to Presidio.
BODY_LIMITS: list[tuple[str, int]] = [
    ("/v1/documents", 2_000_000),
    ("/v1/admin", 64_000),
    ("/v1/query", 16_000),
]
DEFAULT_BODY_LIMIT = 16_000


class EdgeMiddleware:
    """Request IDs + body size limits, as plain ASGI.

    Plain ASGI rather than BaseHTTPMiddleware: that wrapper buffers streaming responses,
    which would hold the whole SSE answer in memory and defeat streaming.
    The size limit counts bytes as they arrive, so a client that lies in Content-Length
    (or sends chunked with no length) still can't push an unbounded body into memory.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = rid
        path: str = scope["path"]
        limit = next((n for prefix, n in BODY_LIMITS if path.startswith(prefix)),
                     DEFAULT_BODY_LIMIT)

        async def send_with_id(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                msg.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(msg)

        for k, v in scope.get("headers", []):
            if k == b"content-length" and v.isdigit() and int(v) > limit:
                await _too_large(send_with_id)
                return
        seen = 0

        async def counted() -> Message:
            nonlocal seen
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > limit:
                    raise _BodyTooLarge
            return msg

        try:
            await self.app(scope, counted, send_with_id)
        except _BodyTooLarge:
            await _too_large(send_with_id)


class _BodyTooLarge(Exception):
    pass


async def _too_large(send: Send) -> None:
    body = b'{"error":"body_too_large"}'
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


# --- errors ------------------------------------------------------------------------------

def error_response(e: GovRagError) -> JSONResponse:
    headers: dict[str, str] = {}
    if e.retry_after_s is not None:
        headers["Retry-After"] = str(max(1, round(e.retry_after_s)))
    if isinstance(e, Unauthorized):
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse({"error": e.code}, status_code=e.status, headers=headers)


def install_errors(app: FastAPI) -> None:
    @app.exception_handler(GovRagError)
    async def _gov(_r: Request, e: GovRagError) -> JSONResponse:
        return error_response(e)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_r: Request, e: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes the offending input back ("input": ...). Input here
        # can be a document or a question full of PII, so return field locations only.
        fields = [".".join(str(x) for x in err.get("loc", ())) for err in e.errors()]
        return JSONResponse({"error": "bad_request", "fields": fields[:20]}, status_code=422)

    @app.exception_handler(Exception)
    async def _any(r: Request, e: Exception) -> JSONResponse:
        # Type name only: third-party exception messages can carry data.
        log.error("unhandled %s request_id=%s", type(e).__name__,
                  r.scope.get("state", {}).get("request_id"))
        return JSONResponse({"error": "internal"}, status_code=500)


# --- app ---------------------------------------------------------------------------------

def services(request: Request) -> Services:
    svc: Services = request.app.state.svc
    return svc


def principal(request: Request) -> Principal:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Unauthorized("missing_token")
    return services(request).verifier.verify(token.strip())


def request_id(request: Request) -> uuid.UUID:
    return uuid.UUID(request.scope["state"]["request_id"])


def create_app(build: Callable[[], Awaitable[Services]] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.svc = await (build() if build else build_services(get_settings()))
        try:
            yield
        finally:
            await app.state.svc.close()

    app = FastAPI(title="GovRAG", lifespan=lifespan, docs_url=None, redoc_url=None)
    install_errors(app)
    from govrag.api import admin, documents, health, query

    for mod in (query, documents, admin, health):
        app.include_router(mod.router)
    app.add_middleware(EdgeMiddleware)
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="127.0.0.1", port=8080, log_level="info",
                access_log=False)  # access logs would include query strings
