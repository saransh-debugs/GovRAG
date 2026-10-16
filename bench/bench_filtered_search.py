"""Filtered HNSW search under RLS: does it return k rows, how good is recall, how fast is it.

The failure this checks for: HNSW returns the ef_search nearest rows, then RLS filters them.
For a role that can see 1% of rows, ~1 of 100 candidates survives, so a plain query returns
fewer than k results. pgvector >= 0.8 iterative scans keep scanning until k rows pass.

Uses its own database (govrag_bench) so it never touches dev data.

    python -m bench.bench_filtered_search --rows 100000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
import uuid
from pathlib import Path

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from govrag import migrate
from govrag.config import get_settings

DIM = 384
BENCH_DB = "govrag_bench"
# (tenant, share of all rows). Small tenants are the hard case for a shared index.
TENANTS = [("big", 0.90), ("mid", 0.09), ("small", 0.01)]
RARE_SHARE = 0.01  # inside "big", 1% of chunks are tagged "rare"


async def _setup(rows: int, seed: int) -> None:
    s = get_settings()
    admin = await asyncpg.connect(s.dsn(s.db_owner_user, s.db_owner_password))
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {BENCH_DB} WITH (FORCE)")
        await admin.execute(f"CREATE DATABASE {BENCH_DB}")
    finally:
        await admin.close()

    s.db_name = BENCH_DB
    conn = await asyncpg.connect(s.dsn(s.db_owner_user, s.db_owner_password))
    try:
        await migrate.apply(conn, s)
        await register_vector(conn)
        # Bulk load: building HNSW once at the end is ~10x faster than per-row inserts.
        await conn.execute("SET statement_timeout = 0")
        await conn.execute("DROP INDEX core.chunks_embedding_hnsw")
        rng = np.random.default_rng(seed)
        # Clustered data, closer to real embeddings than uniform noise: 200 topic centers.
        centers = rng.standard_normal((200, DIM)).astype(np.float32)
        for tenant, share in TENANTS:
            n = int(rows * share)
            await conn.execute("INSERT INTO core.tenants (id) VALUES ($1)", tenant)
            doc = uuid.uuid4()
            await conn.execute(
                "INSERT INTO core.documents (tenant_id, id, title, allowed_roles, created_by)"
                " VALUES ($1, $2, 'bench', ARRAY['all','rare'], 'bench')",
                tenant,
                doc,
            )
            vecs = centers[rng.integers(0, 200, n)] + 0.5 * rng.standard_normal((n, DIM))
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            rare = rng.random(n) < RARE_SHARE
            records = [
                (
                    tenant,
                    uuid.uuid4(),
                    doc,
                    i,
                    "x",
                    vecs[i].astype(np.float32),
                    ["all", "rare"] if rare[i] else ["all"],
                )
                for i in range(n)
            ]
            await conn.copy_records_to_table(
                "chunks",
                schema_name="core",
                records=records,
                columns=[
                    "tenant_id", "id", "document_id", "ord",
                    "text_redacted", "embedding", "allowed_roles",
                ],
            )
        t0 = time.perf_counter()
        await conn.execute(
            "CREATE INDEX chunks_embedding_hnsw ON core.chunks"
            " USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
        print(f"hnsw build {time.perf_counter() - t0:.1f}s")
        await conn.execute("ANALYZE core.chunks")
    finally:
        await conn.close()


SEARCH = """
WITH c AS MATERIALIZED (
  SELECT id, embedding <=> $1 AS dist
  FROM core.chunks
  WHERE tenant_id = $2 AND allowed_roles && $3::text[]
  ORDER BY embedding <=> $1
  LIMIT $4
)
SELECT id, dist FROM c ORDER BY dist
"""


async def _run(queries: int, k: int, seed: int) -> dict[str, object]:
    s = get_settings()
    s.db_name = BENCH_DB
    conn = await asyncpg.connect(s.dsn("govrag_app", s.db_app_password))
    owner = await asyncpg.connect(s.dsn(s.db_owner_user, s.db_owner_password))
    await register_vector(conn)
    await register_vector(owner)
    # Same centers as _setup (first draw from the same seed). Queries must look like the
    # data: a uniform random query in 384-d is ~equally far from everything, so "nearest"
    # is noise and recall numbers mean nothing.
    centers = np.random.default_rng(seed).standard_normal((200, DIM)).astype(np.float32)
    rng = np.random.default_rng(seed + 1)
    results: dict[str, object] = {}
    cases = [("big", ["all"]), ("big", ["rare"]), ("mid", ["all"]), ("small", ["all"])]
    try:
        for tenant, roles in cases:
            for mode in ("off", "relaxed_order"):
                lat, got, recall = [], [], []
                for _ in range(queries):
                    q = centers[rng.integers(0, 200)] + 0.5 * rng.standard_normal(DIM)
                    q = (q / np.linalg.norm(q)).astype(np.float32)
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT set_config('app.tenant', $1, true),"
                            " set_config('app.roles', $2, true),"
                            " set_config('hnsw.iterative_scan', $3, true),"
                            " set_config('hnsw.ef_search', '100', true),"
                            " set_config('hnsw.max_scan_tuples', '20000', true)",
                            tenant,
                            "{" + ",".join(roles) + "}",
                            mode,
                        )
                        t0 = time.perf_counter()
                        rows = await conn.fetch(SEARCH, q, tenant, roles, k)
                        lat.append((time.perf_counter() - t0) * 1000)
                    got.append(len(rows))
                    # Exact answer: owner bypasses RLS, so filter explicitly, no index.
                    async with owner.transaction():
                        await owner.execute("SET LOCAL enable_indexscan = off")
                        exact = await owner.fetch(SEARCH, q, tenant, roles, k)
                    ids = {r["id"] for r in exact}
                    recall.append(len(ids & {r["id"] for r in rows}) / max(len(ids), 1))
                key = f"{tenant}/{roles[0]}/iterative={mode}"
                results[key] = {
                    "rows_returned_mean": statistics.mean(got),
                    "queries_short_of_k": sum(1 for g in got if g < k),
                    "recall_at_k": round(statistics.mean(recall), 3),
                    "p50_ms": round(statistics.median(lat), 2),
                    "p99_ms": round(sorted(lat)[int(0.99 * (len(lat) - 1))], 2),
                }
                print(key, results[key])

        # Show the plan for the hardest case, as the app login (RLS on).
        q = (centers[0] / np.linalg.norm(centers[0])).astype(np.float32)
        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.tenant','big',true), set_config('app.roles','{rare}',true),"
                " set_config('hnsw.iterative_scan','relaxed_order',true)"
            )
            plan = await conn.fetch("EXPLAIN (ANALYZE, BUFFERS) " + SEARCH, q, "big", ["rare"], k)
        # Drop the 384-float literal from the plan text so it stays readable.
        results["plan_big_rare"] = [re.sub(r"'\[[^']*\]'", "'<q>'", r[0]) for r in plan]
        print("\n".join(results["plan_big_rare"]))  # type: ignore[arg-type]
    finally:
        await conn.close()
        await owner.close()
    return results


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-setup", action="store_true")
    a = ap.parse_args()
    if not a.skip_setup:
        t0 = time.perf_counter()
        await _setup(a.rows, a.seed)
        print(f"setup {a.rows} rows in {time.perf_counter() - t0:.1f}s")
    res = await _run(a.queries, a.k, a.seed)
    out = Path(__file__).parent / "results" / "filtered_search.json"
    out.write_text(json.dumps({"rows": a.rows, "k": a.k, "results": res}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(_main())
