"""POST /v1/documents, DELETE /v1/documents/{id}."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from govrag.api.app import principal, request_id, services
from govrag.auth import Principal
from govrag.ingest import DocumentIn, SectionIn

router = APIRouter()


class SectionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    allowed_roles: list[str] | None = None


class DocumentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    allowed_roles: list[str] = Field(min_length=1, max_length=32)
    sections: list[SectionBody] = Field(min_length=1, max_length=1000)


@router.post("/v1/documents", status_code=201)
async def create_document(
    body: DocumentBody,
    request: Request,
    p: Annotated[Principal, Depends(principal)],
) -> dict[str, Any]:
    svc = services(request)
    active = await svc.policies.get(p.tenant_id)
    res = await svc.ingestor.ingest(
        p,
        DocumentIn(
            title=body.title,
            allowed_roles=tuple(body.allowed_roles),
            sections=tuple(
                SectionIn(s.text, tuple(s.allowed_roles) if s.allowed_roles else None)
                for s in body.sections
            ),
        ),
        extra_redact=active.policy.extra_redact,
        redaction_rev=active.version,
        request_id=request_id(request),
    )
    # Counts per PII type only. Never the values, never the redacted text.
    return {
        "document_id": str(res.document_id),
        "chunks": len(res.chunk_ids),
        "pii_counts": res.pii_counts,
    }


@router.delete("/v1/documents/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    request: Request,
    p: Annotated[Principal, Depends(principal)],
) -> dict[str, Any]:
    erased = await services(request).ingestor.delete(
        p, document_id, request_id=request_id(request)
    )
    return {"document_id": str(document_id), "erased_tokens": erased}
