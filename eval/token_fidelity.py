"""Do real models copy placeholder tokens exactly? (Real Vertex calls, a few cents.)

A token the model mangles can't be rehydrated, so the user sees an alias where a name
should be. A token the model invents is harmless (it becomes an alias) but signals the
model is guessing. Counted per model:
  exact      - [TYPE_<16 hex>] that appears in the context
  bare       - TYPE_<16 hex> without brackets (rehydrator handles these too)
  invented   - well-formed token not in the context
  mangled    - token-ish text that no regex accepts (e.g. truncated hex): lost

Uses the synthetic corpus's gold spans to redact, so no Presidio is needed.

    python -m eval.token_fidelity --docs 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from pathlib import Path

from eval.synth_corpus import build
from govrag import prompt
from govrag.config import get_settings
from govrag.llm import LlmError, vertex_factory
from govrag.pii.redact import redact
from govrag.pii.tokens import TOKEN_RE

BARE = re.compile(r"(?<![\w\[])([A-Z][A-Z0-9_]{0,40})_([0-9a-f]{16})(?![\w\]])")
LOOSE = re.compile(r"[A-Z]{3,}[A-Z_]*_[0-9a-fA-F]{4,}")
QUESTION = "List every person mentioned and what each one did. Mention each person."


async def one(model: object, user: str) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    first = 0.0
    parts: list[str] = []
    async for piece in model.stream(system=prompt.SYSTEM, user=user, max_tokens=400):  # type: ignore[attr-defined]
        if not parts:
            first = time.perf_counter() - t0
        parts.append(piece)
    return "".join(parts), first, time.perf_counter() - t0


async def _collect(n_docs: int, concurrency: int) -> dict[str, object]:
    s = get_settings()
    make = vertex_factory(project=s.vertex_project, timeout_s=s.llm_timeout_s,
                          model_ids=s.vertex_model_ids, locations=s.vertex_locations)
    key = os.urandom(32)
    users: list[tuple[str, set[str]]] = []
    for d in build(n_docs, seed=11):
        red = [redact(t, [(a, b, ty, 1.0) for a, b, ty in g], key)
               for t, g in zip(d.sections, d.gold, strict=True)]
        chunks = [prompt.ContextChunk(i + 1, "Doc", r.text) for i, r in enumerate(red)]
        user, _ = prompt.build(QUESTION, chunks, max_chars=s.max_context_chars)
        ctx = {f"{m.group(1)}_{m.group(2)}" for r in red for m in TOKEN_RE.finditer(r.text)}
        users.append((user, ctx))

    sem = asyncio.Semaphore(concurrency)
    results: dict[str, object] = {}
    for name in s.vertex_model_ids:
        results[name] = await eval_model(make(name, "us"), users, sem)
        print(name, results[name])
    return results


async def eval_model(
    model: object, users: list[tuple[str, set[str]]], sem: asyncio.Semaphore
) -> dict[str, object]:
    counts = {"exact": 0, "bare": 0, "invented": 0, "mangled": 0, "errors": 0}
    firsts: list[float] = []
    totals: list[float] = []
    person_answers = 0

    async def run(user: str, ctx: set[str]) -> None:
        nonlocal person_answers
        async with sem:
            try:
                text, first, total = await one(model, user)
            except LlmError:
                counts["errors"] += 1
                return
        firsts.append(first)
        totals.append(total)
        strict = [f"{m.group(1)}_{m.group(2)}" for m in TOKEN_RE.finditer(text)]
        bare = [f"{m.group(1)}_{m.group(2)}" for m in BARE.finditer(text)]
        counts["exact"] += sum(t in ctx for t in strict)
        counts["bare"] += sum(t in ctx for t in bare)
        counts["invented"] += sum(t not in ctx for t in strict + bare)
        ok = [m.span() for m in TOKEN_RE.finditer(text)] + [m.span() for m in BARE.finditer(text)]
        counts["mangled"] += sum(
            1 for m in LOOSE.finditer(text) if not any(a <= m.start() < b for a, b in ok)
        )
        person_answers += any(t.startswith("PERSON_") for t in strict + bare)

    await asyncio.gather(*(run(u, c) for u, c in users))
    emitted = counts["exact"] + counts["bare"] + counts["invented"] + counts["mangled"]
    return {
        **counts,
        "calls": len(users),
        "answers_naming_a_person": person_answers,
        "usable_rate": round((counts["exact"] + counts["bare"]) / max(emitted, 1), 4),
        "first_token_p50_s": round(statistics.median(firsts), 2) if firsts else None,
        "total_p95_s": round(sorted(totals)[int(0.95 * (len(totals) - 1))], 2) if totals else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    res = asyncio.run(_collect(a.docs, a.concurrency))
    out = Path(__file__).resolve().parent.parent / "bench" / "results" / "token_fidelity.json"
    out.write_text(json.dumps({"docs": a.docs, "region": "us", "results": res}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
