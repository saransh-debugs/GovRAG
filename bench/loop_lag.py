"""Event loop lag while CPU work runs elsewhere: thread vs process, plus an idle control.

A ticker sleeps 5 ms in a loop and records how late it wakes up. Reports p50/p99/max.

Why the idle control matters: on macOS, a process that is doing *nothing* still sees
rare 60-110 ms timer delays. So the single worst sample says nothing about our code;
p99 does. Real blocking (e.g. pure-Python chunking holding the GIL) repeats on every
call and moves p99.

    python -m bench.loop_lag
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from govrag.chunking import Chunker

TEXTS = ["Compensation review. [PERSON_3f9a1c2e4b7d9a01] gets a raise and more words. " * 6] * 120


async def measure(fn: Callable[[], Awaitable[Any]]) -> dict[str, float]:
    lags: list[float] = []
    done = False

    async def tick() -> None:
        while not done:
            t = time.perf_counter()
            await asyncio.sleep(0.005)
            lags.append(time.perf_counter() - t - 0.005)

    tk = asyncio.create_task(tick())
    t0 = time.perf_counter()
    await fn()
    total = time.perf_counter() - t0
    done = True
    await tk
    lags.sort()
    n = len(lags)
    return {"total_ms": round(total * 1000), "p50_ms": round(lags[n // 2] * 1000, 1),
            "p99_ms": round(lags[int(n * 0.99)] * 1000, 1), "max_ms": round(lags[-1] * 1000, 1),
            "samples": n}


async def _main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    from fastembed import TextEmbedding

    model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=2)
    chunker = Chunker()
    big = " ".join(TEXTS) * 3
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(1) as ex:
        await loop.run_in_executor(ex, time.sleep, 0)
        cases: dict[str, Callable[[], Awaitable[Any]]] = {
            "idle (control)": lambda: asyncio.sleep(3),
            "embed in thread": lambda: asyncio.to_thread(
                lambda: list(model.passage_embed(TEXTS, batch_size=64))),
            "chunk in thread": lambda: asyncio.to_thread(
                lambda: [chunker.split(big) for _ in range(5)]),
            "chunk in process": lambda: loop.run_in_executor(ex, _chunk_many, big),
        }
        out: dict[str, list[dict[str, float]]] = {}
        for _ in range(2):
            for name, fn in cases.items():
                r = await measure(fn)
                out.setdefault(name, []).append(r)
                print(f"{name:18} {r}")
    path = Path(__file__).parent / "results" / "loop_lag.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


def _chunk_many(text: str) -> int:
    c = Chunker()
    return sum(len(c.split(text)) for _ in range(5))


if __name__ == "__main__":
    asyncio.run(_main())
