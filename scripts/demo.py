"""End-to-end demo against a running server (python -m govrag.api.app).

Seeds two tenants, uploads documents, then asks the same question as different users,
edits the policy, and asks again. Prints what each user actually gets back.

    python -m scripts.demo            # server on http://127.0.0.1:8080
"""

from __future__ import annotations

import asyncio
import json

import asyncpg
import httpx

from govrag import migrate
from govrag.config import get_settings
from govrag.keys import MasterKey, provision_tenant
from scripts.mint_jwt import token

BASE = "http://127.0.0.1:8080"
Q = "Who got a raise, what is their SSN, and who approved it?"
Q2 = "Who runs the Q3 review cycle, and when does it close?"


async def provision() -> None:
    s = get_settings()
    conn = await asyncpg.connect(s.dsn(s.db_owner_user, s.db_owner_password))
    try:
        await migrate.apply(conn, s)
        for t in ("acme", "globex"):
            if not await conn.fetchval("SELECT 1 FROM core.tenants WHERE id = $1", t):
                await provision_tenant(conn, MasterKey(s.master_keys.get_secret_value()), t)
    finally:
        await conn.close()


def h(tenant: str, user: str, *roles: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token(tenant, user, list(roles))}"}


def show(label: str, r: httpx.Response) -> None:
    body = r.json()
    print(f"\n--- {label}  [HTTP {r.status_code}]")
    if "answer" in body:
        print(body["answer"].strip())
        print(f"(model={body['model']}, policy v{body['policy_version']}, "
              f"unmasked={body['unmasked_types']}, masked={body['masked_types']}, "
              f"sources={len(body['citations'])})")
    else:
        print(json.dumps(body))


async def main() -> None:
    await provision()
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as c:
        admin = h("acme", "u-admin", "policy_admin")
        cur = (await c.get("/v1/admin/policy", headers=admin)).json()
        r = await c.put("/v1/admin/policy", headers=admin, json={
            "expected_version": cur["version"],
            "policy": {"unmask": {"hr": ["PERSON", "US_SSN"]}},
        })
        print("policy:", r.json()["version"], r.json()["policy"]["unmask"])

        ing = h("acme", "svc-ingest", "ingest")
        await c.post("/v1/documents", headers=ing, json={
            "title": "Q3 compensation review", "allowed_roles": ["hr", "intern"],
            "sections": [
                {"text": "Margaret Okafor (SSN 078-05-1120) gets a 12% raise, approved by "
                         "Daniel Whitfield.", "allowed_roles": ["hr"]},
                {"text": "The Q3 review cycle closes on Sept 30. Daniel Whitfield runs it."},
            ]})
        await c.post("/v1/documents", headers=h("globex", "svc-ingest", "ingest"), json={
            "title": "Globex payroll", "allowed_roles": ["hr", "intern"],
            "sections": [{"text": "Globex: Priyanka Raghunathan gets a 30% raise."}]})

        show("acme hr", await c.post("/v1/query", headers=h("acme", "u-hr", "hr"),
                                     json={"question": Q}))
        show("acme intern", await c.post("/v1/query", headers=h("acme", "u-intern", "intern"),
                                         json={"question": Q}))
        show("globex hr", await c.post("/v1/query", headers=h("globex", "u-g", "hr"),
                                       json={"question": Q}))
        show("acme intern: who runs the review?",
             await c.post("/v1/query", headers=h("acme", "u-intern", "intern"),
                          json={"question": Q2}))

        cur = (await c.get("/v1/admin/policy", headers=admin)).json()
        await c.put("/v1/admin/policy", headers=admin, json={
            "expected_version": cur["version"],
            "policy": {"unmask": {"hr": ["PERSON", "US_SSN"], "intern": ["PERSON"]}},
        })
        print("\n(policy edited: intern may now see PERSON)")
        show("acme intern: who runs the review? (next request)",
             await c.post("/v1/query", headers=h("acme", "u-intern", "intern"),
                          json={"question": Q2}))
        show("no token", await c.post("/v1/query", json={"question": Q}))


if __name__ == "__main__":
    asyncio.run(main())
