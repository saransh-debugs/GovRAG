"""What does the security layer cost per query? Load test with a zero-latency fake LLM.

Everything is real except the LLM: Presidio on the question (query lane), local
embeddings, pgvector search under RLS, two audit inserts, vault decrypt, rehydration.
Per-stage times come from the same OpenTelemetry spans production would export.

Also measures ingest throughput (chunks/s) with 1 vs 2 Presidio workers.

Own database (govrag_loadtest), so dev data is never touched.

    python -m bench.bench_query
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import asyncpg
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from eval.synth_corpus import build
from govrag import migrate
from govrag.api.app import Services, build_services
from govrag.auth import JwtVerifier, Principal
from govrag.config import get_settings
from govrag.errors import Overloaded
from govrag.ingest import DocumentIn, SectionIn
from govrag.keys import MasterKey, provision_tenant
from govrag.llm import FakeLLM
from govrag.pii.analyzer import PiiAnalyzer
from govrag.policy.model import Policy

DB = "govrag_loadtest"
T = "load"
QUESTIONS = ["who approved the budget", "what is the migration schedule", "offsite contact",
             "who joined the platform team", "phone number for follow-ups", "badge request"]
STAGES = ["policy", "redact_question", "embed", "retrieve", "audit.retrieve",
          "llm.first_token", "vault.wait", "audit.rehydrate"]


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))] * 1000, 1)


async def setup() -> None:
    s = get_settings()
    admin = await asyncpg.connect(s.dsn(s.db_owner_user, s.db_owner_password))
    await admin.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
    await admin.execute(f"CREATE DATABASE {DB}")
    await admin.close()
    s.db_name = DB
    conn = await asyncpg.connect(s.dsn(s.db_owner_user, s.db_owner_password))
    await migrate.apply(conn, s)
    await provision_tenant(conn, MasterKey(s.master_keys.get_secret_value()), T)
    await conn.close()


async def ingest_rate(svc: Services, docs: list[DocumentIn], parallel: int) -> dict[str, float]:
    """`parallel` concurrent ingests (the Ingestor's own cap is max_inflight_ingests=2).
    One small document fits in one Presidio batch, so a second worker only helps when two
    documents are in flight at once."""
    p = Principal(T, "svc", frozenset({"ingest"}))
    sem = asyncio.Semaphore(parallel)

    async def one(d: DocumentIn) -> int:
        async with sem:
            return len((await svc.ingestor.ingest(p, d)).chunk_ids)

    t0 = time.perf_counter()
    chunks = sum(await asyncio.gather(*(one(d) for d in docs)))
    dt = time.perf_counter() - t0
    return {"docs": len(docs), "chunks": chunks, "seconds": round(dt, 1),
            "chunks_per_s": round(chunks / dt, 1)}


async def _collect() -> dict[str, object]:
    await setup()
    s = get_settings()
    s.db_name = DB
    s.otlp_endpoint = ""
    s.max_inflight_queries = 256
    s.max_inflight_per_tenant = 256
    mem = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(mem))
    trace.set_tracer_provider(tp)
    fake = FakeLLM(mode="script", script=lambda _s, u: "Answer based on [doc 1].")
    out: dict[str, object] = {}

    corpus = build(120, seed=3)
    docs = [DocumentIn("Memo", ("staff",), tuple(SectionIn(x) for x in d.sections))
            for d in corpus]

    for workers in (1, 2):
        a = PiiAnalyzer(model=s.spacy_model, ingest_workers=workers, query_workers=1)
        await a.start()
        svc = await build_services(
            s, verifier=JwtVerifier(b"", issuer="x", audience="x"),
            llm_factory=lambda _n, _r: fake, analyzer=a, background=False,
        )
        try:
            # Ingest one batch per worker count; the second batch also grows the corpus.
            batch = docs[:60] if workers == 1 else docs[60:]
            mem.clear()
            row = await ingest_rate(svc, batch, parallel=2)
            stage_s: dict[str, float] = defaultdict(float)
            for sp in mem.get_finished_spans():
                if sp.name in ("pii.analyze", "redact_and_chunk", "embed", "ingest") \
                        and sp.end_time and sp.start_time:
                    stage_s[sp.name] += (sp.end_time - sp.start_time) / 1e9
            # Summed over documents. With 2 in flight, the sum can exceed wall time.
            row["stage_seconds_summed"] = {k: round(v, 1) for k, v in stage_s.items()}
            out[f"ingest_{workers}_workers"] = row
            print(f"ingest {workers} worker(s):", out[f"ingest_{workers}_workers"])
            if workers == 2:
                await svc.store.put(Principal(T, "a", frozenset({"policy_admin"})),
                                    Policy.model_validate({"unmask": {"staff": ["PERSON"]}}),
                                    expected_version=0)
                p = Principal(T, "u1", frozenset({"staff"}))
                for _ in range(10):  # warm caches (policy, keys, pools)
                    await svc.query.answer(p, QUESTIONS[0])
                for conc, n in ((1, 120), (8, 240), (32, 320)):
                    mem.clear()
                    lat: list[float] = []
                    shed = 0
                    sem = asyncio.Semaphore(conc)

                    async def one(i: int, sem: asyncio.Semaphore = sem, lat: list[float] = lat,
                                  svc: Services = svc, p: Principal = p) -> int:
                        async with sem:
                            t = time.perf_counter()
                            try:
                                await svc.query.answer(p, QUESTIONS[i % len(QUESTIONS)])
                            except Overloaded:
                                return 1
                            lat.append(time.perf_counter() - t)
                            return 0

                    t0 = time.perf_counter()
                    shed = sum(await asyncio.gather(*(one(i) for i in range(n))))
                    wall = time.perf_counter() - t0
                    by: dict[str, list[float]] = defaultdict(list)
                    for sp in mem.get_finished_spans():
                        if sp.name in STAGES and sp.end_time and sp.start_time:
                            by[sp.name].append((sp.end_time - sp.start_time) / 1e9)
                    row = {
                        "requests": n, "shed": shed, "qps": round(len(lat) / wall, 1),
                        "p50_ms": pct(lat, 0.5), "p99_ms": pct(lat, 0.99),
                        "stages_p50_ms": {k: pct(by[k], 0.5) for k in STAGES if by[k]},
                        "stages_p99_ms": {k: pct(by[k], 0.99) for k in STAGES if by[k]},
                    }
                    out[f"query_concurrency_{conc}"] = row
                    print(f"concurrency {conc}:", {k: row[k] for k in
                                                   ("qps", "p50_ms", "p99_ms", "shed")})
                    print("   p50 by stage:", row["stages_p50_ms"])
        finally:
            await svc.close()
            a.close()
    return out


def main() -> None:
    res = asyncio.run(_collect())
    path = Path(__file__).resolve().parent / "results" / "query_overhead.json"
    path.write_text(json.dumps({
        "hardware": "Intel i7-9750H (6c/12t), 16 GB, macOS; Postgres 17 + pgvector 0.8.1 in "
                    "Docker; fake LLM (zero latency); 1 Presidio query worker",
        **res,
    }, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
