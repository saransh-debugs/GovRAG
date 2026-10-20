"""Smoke-test the real Vertex models through GovRAG's own LLM classes.

One tiny streamed call per (model, region). Reports first-token latency, total time,
and the exact error code if a model isn't usable yet (e.g. not enabled in Model Garden).
Costs a fraction of a cent.

    python -m scripts.vertex_check
"""

from __future__ import annotations

import asyncio
import time

from govrag.config import get_settings
from govrag.llm import LlmError, vertex_factory

PROMPT = "Reply with exactly: [PERSON_3f9a1c2e4b7d9a01] approved it."


async def check(make: object, name: str, region: str) -> None:
    model = make(name, region)  # type: ignore[operator]
    t0 = time.perf_counter()
    first = None
    out: list[str] = []
    try:
        async for piece in model.stream(system="Follow the instruction exactly.", user=PROMPT,
                                        max_tokens=40):
            if first is None:
                first = time.perf_counter() - t0
            out.append(piece)
    except LlmError as e:
        cause = e.__cause__
        print(f"{name:24} {region:3} FAIL {e.code}: {type(cause).__name__}: {str(cause)[:160]}")
        return
    total = time.perf_counter() - t0
    text = "".join(out).strip()
    exact = "[PERSON_3f9a1c2e4b7d9a01]" in text
    print(f"{name:24} {region:3} OK first_token={first or 0:.2f}s total={total:.2f}s "
          f"token_copied={exact} reply={text[:60]!r}")


async def main() -> None:
    s = get_settings()
    if not s.vertex_project:
        raise SystemExit("set GOVRAG_VERTEX_PROJECT")
    make = vertex_factory(project=s.vertex_project, timeout_s=s.llm_timeout_s,
                          model_ids=s.vertex_model_ids, locations=s.vertex_locations)
    for name in s.vertex_model_ids:
        for region in ("us", "eu"):
            await check(make, name, region)


if __name__ == "__main__":
    asyncio.run(main())
