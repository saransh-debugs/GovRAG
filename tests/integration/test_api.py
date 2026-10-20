"""HTTP surface: auth, JSON + SSE, limits, error shapes, admin endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

import asyncpg
import httpx
import pytest

from govrag.api.app import build_services, create_app
from govrag.config import Settings
from govrag.embed import HashEmbedder
from govrag.keys import MasterKey, provision_tenant
from govrag.limits import Admission
from govrag.pii.analyzer import PiiAnalyzer
from tests.conftest import JwtKit
from tests.security.test_query_security import FakeBox


@pytest.fixture(scope="module")
async def env(
    settings: Settings, owner: asyncpg.Connection, master: MasterKey, analyzer: PiiAnalyzer,
    jwtkit: JwtKit,
) -> AsyncIterator[tuple[httpx.AsyncClient, str, object]]:
    t = f"api-{os.urandom(3).hex()}"
    await provision_tenant(owner, master, t)
    box = FakeBox()

    async def build() -> object:
        return await build_services(settings, verifier=jwtkit.verifier, llm_factory=box.factory,
                                    embedder=HashEmbedder(), analyzer=analyzer, background=False)

    app = create_app(build)  # type: ignore[arg-type]
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            yield client, t, app


def auth(jwtkit: JwtKit, t: str, *roles: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {jwtkit.token(tenant=t, user='u1', roles=list(roles))}"}


async def test_health(env: tuple[httpx.AsyncClient, str, object]) -> None:
    c, _, app = env
    assert (await c.get("/healthz")).json() == {"ok": True}
    # Not ready until the policy listener is connected (it connects in the background).
    await asyncio.wait_for(app.state.svc.policies.listening.wait(), 5)  # type: ignore[attr-defined]
    r = await c.get("/readyz")
    assert r.status_code == 200 and r.json()["db"] is True and "x-request-id" in r.headers


async def test_auth_required(env: tuple[httpx.AsyncClient, str, object]) -> None:
    c, _, _ = env
    for headers in ({}, {"Authorization": "Bearer junk"}, {"Authorization": "Basic abc"}):
        r = await c.post("/v1/query", json={"question": "hi"}, headers=headers)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
        assert set(r.json()) == {"error"}


async def test_ingest_then_query_json_and_stream(
    env: tuple[httpx.AsyncClient, str, object], jwtkit: JwtKit
) -> None:
    c, t, _ = env
    r = await c.post("/v1/documents", headers=auth(jwtkit, t, "intern"), json={
        "title": "x", "allowed_roles": ["hr"], "sections": [{"text": "x"}]})
    assert r.status_code == 403
    r = await c.post("/v1/documents", headers=auth(jwtkit, t, "ingest"), json={
        "title": "Handbook", "allowed_roles": ["hr", "intern"],
        "sections": [{"text": "Vacation policy: Margaret Okafor approves 20 days of leave."}]})
    assert r.status_code == 201 and r.json()["pii_counts"] == {"PERSON": 1}
    doc_id = r.json()["document_id"]

    r = await c.post("/v1/query", headers=auth(jwtkit, t, "intern"),
                     json={"question": "vacation policy leave"})
    body = r.json()
    assert r.status_code == 200 and body["citations"][0]["document_id"] == doc_id
    assert "Margaret" not in body["answer"] and "[PERSON 1]" in body["answer"]
    assert body["request_id"] == r.headers["x-request-id"]

    async with c.stream("POST", "/v1/query/stream", headers=auth(jwtkit, t, "intern"),
                        json={"question": "vacation policy leave"}) as s:
        assert s.status_code == 200 and s.headers["content-type"].startswith("text/event-stream")
        raw = "".join([x async for x in s.aiter_text()])
    events = [(blk.split("\n")[0][7:], json.loads(blk.split("\n")[1][6:]))
              for blk in raw.strip().split("\n\n")]
    assert events[-1][0] == "done" and all(e == "delta" for e, _ in events[:-1])
    assert "".join(d["text"] for _, d in events[:-1]) == body["answer"]

    r = await c.delete(f"/v1/documents/{doc_id}", headers=auth(jwtkit, t, "ingest"))
    assert r.status_code == 200 and r.json()["erased_tokens"] == 1


async def test_limits_and_error_shapes(
    env: tuple[httpx.AsyncClient, str, object], jwtkit: JwtKit
) -> None:
    c, t, _ = env
    h = auth(jwtkit, t, "intern")
    r = await c.post("/v1/query", headers=h, content=b'{"question":"' + b"a" * 20_000 + b'"}')
    assert r.status_code == 413 and r.json() == {"error": "body_too_large"}
    # Over the field limit but under the body limit: a 422 that must not echo the input.
    secret = "078-05-1120"
    r = await c.post("/v1/query", headers=h, json={"question": f"{secret} " + "b" * 2100})
    assert r.status_code == 422 and secret not in r.text and r.json()["error"] == "bad_request"
    r = await c.post("/v1/query", headers=h, json={"question": "x", "surprise": True})
    assert r.status_code == 422


async def test_overload_is_503_with_retry_after(
    env: tuple[httpx.AsyncClient, str, object], jwtkit: JwtKit
) -> None:
    c, t, app = env
    q = app.state.svc.query  # type: ignore[attr-defined]
    real = q._admission
    q._admission = Admission(global_max=0, per_tenant_max=0)
    try:
        for path in ("/v1/query", "/v1/query/stream"):
            r = await c.post(path, headers=auth(jwtkit, t, "intern"), json={"question": "hi"})
            assert r.status_code == 503 and r.headers["retry-after"] == "1"
            assert r.json() == {"error": "tenant_busy"}
    finally:
        q._admission = real


async def test_admin_policy_endpoints(
    env: tuple[httpx.AsyncClient, str, object], jwtkit: JwtKit
) -> None:
    c, t, _ = env
    admin, user = auth(jwtkit, t, "policy_admin"), auth(jwtkit, t, "hr")
    assert (await c.get("/v1/admin/policy", headers=user)).status_code == 403
    r = await c.get("/v1/admin/policy", headers=admin)
    assert r.json()["version"] == 0
    new = {"unmask": {"hr": ["PERSON"]}}
    r = await c.put("/v1/admin/policy", headers=admin,
                    json={"expected_version": 0, "policy": new})
    assert r.status_code == 200 and r.json()["version"] == 1
    r = await c.put("/v1/admin/policy", headers=admin,
                    json={"expected_version": 0, "policy": new})
    assert r.status_code == 409 and r.json() == {"error": "policy_version_conflict"}
    r = await c.put("/v1/admin/policy", headers=admin,
                    json={"expected_version": 1, "policy": {"llm": {"models": ["gpt-evil"]}}})
    assert r.status_code == 422
    r = await c.post("/v1/admin/policy/dry-run", headers=admin,
                     json={"roles": ["hr"], "question": "vacation"})
    assert r.status_code == 200 and set(r.json()) == {
        "roles", "document_ids", "chunk_ids", "pii_types_present", "pii_types_unmasked"}
    r = await c.post("/v1/admin/policy/rollback", headers=admin,
                     json={"to_version": 1, "expected_version": 1})
    assert r.status_code == 200
    assert (await c.get("/v1/admin/policy/versions", headers=admin)).json()["versions"][0] == {
        "version": 1, "created_by": "u1"}
    assert (await c.post("/v1/admin/shred", headers=admin)).status_code == 403


async def test_revoked_token_gets_401(
    env: tuple[httpx.AsyncClient, str, object], jwtkit: JwtKit
) -> None:
    c, t, _ = env
    victim = auth(jwtkit, t, "intern")  # minted before the revocation
    assert (await c.post("/v1/query", headers=victim, json={"question": "hi"})).status_code == 200
    r = await c.post("/v1/admin/revoke", headers=auth(jwtkit, t, "policy_admin"),
                     json={"user_id": "u1"})
    assert r.status_code == 200
    r = await c.post("/v1/query", headers=victim, json={"question": "hi"})
    assert r.status_code == 401 and r.json() == {"error": "token_revoked"}
