"""Liveness and readiness.

/healthz: the process is up and the event loop answers. No dependencies, so a DB outage
doesn't get every instance restarted by the orchestrator (which wouldn't fix the DB).
/readyz: we can actually serve: DB reachable and the policy listener connected. A load
balancer should stop routing here while this is false.
"""

from __future__ import annotations

from typing import Any

import asyncpg
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from govrag.api.app import services

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    svc = services(request)
    checks: dict[str, bool] = {}
    try:
        async with svc.pools.app.acquire(timeout=1.0) as c:
            checks["db"] = await c.fetchval("SELECT 1", timeout=1.0) == 1
    except (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError):
        checks["db"] = False
    checks["policy_listener"] = svc.policies.listening.is_set()
    ok = all(checks.values())
    return JSONResponse({"ok": ok, **checks}, status_code=200 if ok else 503)
