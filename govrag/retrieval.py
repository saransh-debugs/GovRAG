"""Vector search, filtered by tenant + roles twice: in this SQL, and by RLS underneath.

hnsw.iterative_scan: measured in bench/results/filtered_search.json. Without it, a tenant
holding 9% of rows got fewer than k results on 2 of 50 queries; with it, 0 of 50. The
outer ORDER BY is needed because relaxed_order can return rows slightly out of order.

Why not LlamaIndex's PGVectorStore: our security boundary would become a framework's
metadata-filter code, and it can't set the per-transaction tenant/roles that RLS needs.
We own the SQL and expose it as a LlamaIndex BaseRetriever so it still plugs into
LlamaIndex pipelines.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from govrag.auth import Principal
from govrag.db import Pools, roles_literal, scoped
from govrag.embed import Vec

SEARCH = """
WITH c AS MATERIALIZED (
  SELECT id, document_id, text_redacted, pii_types, embedding <=> $1 AS dist
    FROM core.chunks
   WHERE tenant_id = $2 AND allowed_roles && $3::text[]
   ORDER BY embedding <=> $1
   LIMIT $4
)
SELECT c.id, c.document_id, c.text_redacted, c.pii_types, c.dist, d.title
  FROM c JOIN core.documents d ON d.tenant_id = $2 AND d.id = c.document_id
 ORDER BY c.dist
"""


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    text: str
    pii_types: frozenset[str]
    distance: float


class Retriever:
    def __init__(self, pools: Pools, *, ef_search: int, max_scan_tuples: int) -> None:
        self._pools = pools
        self._extra = {
            "hnsw.iterative_scan": "relaxed_order",
            "hnsw.ef_search": str(ef_search),
            "hnsw.max_scan_tuples": str(max_scan_tuples),
        }

    async def search(self, p: Principal, qvec: Vec, k: int) -> list[Hit]:
        roles = sorted(p.roles)
        roles_literal(roles)  # validates before the DB, again
        async with scoped(
            self._pools.app, p.tenant_id, roles,
            acquire_timeout_s=self._pools.acquire_timeout_s, extra=self._extra,
        ) as c:
            rows = await c.fetch(SEARCH, qvec, p.tenant_id, roles, k)
        return [
            Hit(r["id"], r["document_id"], r["title"], r["text_redacted"],
                frozenset(r["pii_types"]), float(r["dist"]))
            for r in rows
        ]


class GovRetriever(BaseRetriever):
    """LlamaIndex adapter: one instance per request, bound to that request's principal.

    The principal is fixed at construction, not taken from the query, so nothing that
    flows through a LlamaIndex pipeline can change whose permissions the search uses.
    """

    def __init__(self, inner: Retriever, principal: Principal, k: int) -> None:
        super().__init__()
        self._inner = inner
        self._p = principal
        self._k = k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        raise NotImplementedError("async only: a sync DB call would block the event loop")

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if query_bundle.embedding is None:
            raise ValueError("query embedding required")
        import numpy as np

        hits = await self._inner.search(
            self._p, np.asarray(query_bundle.embedding, dtype=np.float32), self._k
        )
        return [
            NodeWithScore(
                node=TextNode(
                    id_=str(h.chunk_id),
                    text=h.text,
                    metadata=_meta(h),
                    excluded_embed_metadata_keys=list(_meta(h)),
                    excluded_llm_metadata_keys=list(_meta(h)),
                ),
                score=1.0 - h.distance,
            )
            for h in hits
        ]


def _meta(h: Hit) -> dict[str, Any]:
    return {
        "document_id": str(h.document_id),
        "title": h.title,
        "pii_types": sorted(h.pii_types),
    }


def hits_from_nodes(nodes: list[NodeWithScore]) -> list[Hit]:
    return [
        Hit(
            chunk_id=uuid.UUID(n.node.node_id),
            document_id=uuid.UUID(n.node.metadata["document_id"]),
            title=n.node.metadata["title"],
            text=n.node.get_content(),
            pii_types=frozenset(n.node.metadata["pii_types"]),
            distance=1.0 - float(n.score or 0.0),
        )
        for n in nodes
    ]
