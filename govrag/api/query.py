"""POST /v1/query (JSON) and POST /v1/query/stream (Server-Sent Events)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from govrag.api.app import principal, request_id, services
from govrag.auth import Principal
from govrag.errors import GovRagError
from govrag.query import AnswerMeta, Delta, Done, Event

router = APIRouter()


class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def _meta(m: AnswerMeta) -> dict[str, Any]:
    d = asdict(m)
    d["request_id"] = str(m.request_id)
    d["citations"] = [
        {**c, "document_id": str(c["document_id"]), "chunk_id": str(c["chunk_id"])}
        for c in d["citations"]
    ]
    return d


@router.post("/v1/query")
async def query(
    body: QueryIn,
    request: Request,
    p: Annotated[Principal, Depends(principal)],
) -> dict[str, Any]:
    ans = await services(request).query.answer(
        p, body.question, top_k=body.top_k, request_id=request_id(request)
    )
    return {"answer": ans.text, **_meta(ans.meta)}


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


@router.post("/v1/query/stream")
async def query_stream(
    body: QueryIn,
    request: Request,
    p: Annotated[Principal, Depends(principal)],
) -> StreamingResponse:
    gen = services(request).query.stream(
        p, body.question, top_k=body.top_k, request_id=request_id(request)
    )
    # Pull the first event *before* committing to a 200. Auth, overload, bad input, policy
    # or audit unavailable then come back as real HTTP status codes (503 + Retry-After
    # etc.), not as a 200 with an error buried in the event stream.
    try:
        first: Event = await anext(gen)
    except BaseException:
        await gen.aclose()
        raise

    async def events() -> AsyncIterator[bytes]:
        try:
            ev: Event | None = first
            while ev is not None:
                if isinstance(ev, Delta):
                    yield _sse("delta", {"text": ev.text})
                elif isinstance(ev, Done):
                    yield _sse("done", _meta(ev.meta))
                ev = await anext(gen, None)
        except GovRagError as e:
            # Mid-stream failure (model died, stall, deadline). Headers are already sent,
            # so the error travels as an event and the client knows the answer is cut off.
            yield _sse("error", {"error": e.code, "request_id": str(request_id(request))})
        finally:
            # Runs on client disconnect too (Starlette cancels this generator), closing the
            # LLM stream so the vendor stops generating.
            await gen.aclose()

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
