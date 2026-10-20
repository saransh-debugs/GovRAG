"""Admin API: policy read/write/rollback/dry-run, crypto-shred."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from govrag.api.app import principal, request_id, services
from govrag.auth import Principal
from govrag.errors import Forbidden
from govrag.policy.model import ActivePolicy, Policy
from govrag.policy.store import ADMIN_ROLE

router = APIRouter(prefix="/v1/admin")


def _admin(p: Principal) -> Principal:
    if not p.has(ADMIN_ROLE):
        raise Forbidden("policy_admin_required")
    return p


def _out(a: ActivePolicy) -> dict[str, Any]:
    return {"version": a.version, "policy": a.policy.model_dump(mode="json")}


@router.get("/policy")
async def get_policy(
    request: Request, p: Annotated[Principal, Depends(principal)]
) -> dict[str, Any]:
    return _out(await services(request).store.get(_admin(p)))


@router.get("/policy/versions")
async def list_versions(
    request: Request, p: Annotated[Principal, Depends(principal)]
) -> dict[str, Any]:
    rows = await services(request).store.versions(p)
    return {"versions": [{"version": v, "created_by": by} for v, by in rows]}


class PutPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_version: int = Field(ge=0)
    policy: Policy


@router.put("/policy")
async def put_policy(
    body: PutPolicy, request: Request, p: Annotated[Principal, Depends(principal)]
) -> dict[str, Any]:
    svc = services(request)
    res = await svc.store.put(p, body.policy, expected_version=body.expected_version,
                              request_id=request_id(request))
    # NOTIFY reaches every instance in milliseconds; this one doesn't even wait for that.
    svc.policies.invalidate(p.tenant_id)
    return _out(res)


class Rollback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to_version: int = Field(ge=1)
    expected_version: int = Field(ge=0)


@router.post("/policy/rollback")
async def rollback(
    body: Rollback, request: Request, p: Annotated[Principal, Depends(principal)]
) -> dict[str, Any]:
    svc = services(request)
    res = await svc.store.rollback(p, body.to_version, expected_version=body.expected_version,
                                   request_id=request_id(request))
    svc.policies.invalidate(p.tenant_id)
    return _out(res)


class DryRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roles: list[str] = Field(min_length=1, max_length=32)
    question: str = Field(min_length=1, max_length=2000)
    policy: Policy | None = None  # None = the live policy
    top_k: int | None = Field(default=None, ge=1, le=20)


@router.post("/policy/dry-run")
async def dry_run(
    body: DryRunBody, request: Request, p: Annotated[Principal, Depends(principal)]
) -> dict[str, Any]:
    svc = services(request)
    policy = body.policy or (await svc.policies.get(_admin(p).tenant_id)).policy
    r = await svc.query.dry_run(p, policy, body.roles, body.question, top_k=body.top_k)
    return {
        "roles": r.roles,
        "document_ids": [str(x) for x in r.document_ids],
        "chunk_ids": [str(x) for x in r.chunk_ids],
        "pii_types_present": r.pii_types_present,
        "pii_types_unmasked": r.pii_types_unmasked,
    }


@router.post("/shred")
async def shred(request: Request, p: Annotated[Principal, Depends(principal)]) -> dict[str, Any]:
    svc = services(request)
    n = await svc.ingestor.shred_tenant_pii(p, svc.master, request_id=request_id(request))
    return {"erased_tokens": n}


class RevokeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=128)


@router.post("/revoke")
async def revoke(
    body: RevokeBody, request: Request, p: Annotated[Principal, Depends(principal)]
) -> dict[str, Any]:
    """Reject every token for this user issued before now. They must log in again."""
    nb = await services(request).revocations.revoke(p, body.user_id)
    return {"user_id": body.user_id, "not_before": nb}
