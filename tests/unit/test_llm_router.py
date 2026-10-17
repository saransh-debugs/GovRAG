"""Fallback, first-token timeout, breaker, mid-stream failure, slot accounting."""

from __future__ import annotations

import pytest

from govrag.errors import Overloaded, Unavailable
from govrag.llm import ChatModel, FakeLLM, LlmRouter


def router(models: dict[str, ChatModel], concurrency: int = 4) -> LlmRouter:
    return LlmRouter(lambda name, _region: models[name], concurrency=concurrency,
                     first_token_timeout_s=0.2, stall_timeout_s=0.2,
                     breaker_failures=2, breaker_cooldown_s=60)


async def collect(r: LlmRouter, names: list[str]) -> tuple[str, str]:
    s = await r.open(names, "us", system="s", user="hello world", max_tokens=10)
    try:
        return s.model, "".join([x async for x in s])
    finally:
        await s.aclose()


async def test_main_ok() -> None:
    assert await collect(router({"a": FakeLLM(), "b": FakeLLM()}), ["a", "b"]) == (
        "a", "hello world")


@pytest.mark.parametrize("fail", ["error", "timeout"])
async def test_falls_back_before_first_token(fail: str) -> None:
    a = FakeLLM(fail=fail)  # type: ignore[arg-type]
    model, text = await collect(router({"a": a, "b": FakeLLM()}), ["a", "b"])
    assert (model, text) == ("b", "hello world")


async def test_all_down_is_unavailable() -> None:
    r = router({"a": FakeLLM(fail="error"), "b": FakeLLM(fail="error")})
    with pytest.raises(Unavailable, match="llm_unavailable"):
        await collect(r, ["a", "b"])


async def test_breaker_skips_dead_model() -> None:
    a = FakeLLM(fail="timeout")
    r = router({"a": a, "b": FakeLLM()})
    for _ in range(2):
        await collect(r, ["a", "b"])  # each pays a 0.2 s timeout on "a"
    calls = len(a.prompts)
    await collect(r, ["a", "b"])
    assert len(a.prompts) == calls  # breaker open: "a" not even tried
    assert not r.breaker("a").allow()


async def test_midstream_failure_surfaces() -> None:
    r = router({"a": FakeLLM(fail="midstream", piece=2), "b": FakeLLM()})
    with pytest.raises(Unavailable, match="llm_failed_midstream"):
        await collect(r, ["a", "b"])


async def test_slot_returned_even_if_never_iterated() -> None:
    r = router({"a": FakeLLM()}, concurrency=1)
    for _ in range(3):
        s = await r.open(["a"], "us", system="s", user="u", max_tokens=1)
        await s.aclose()  # never iterated
    with pytest.raises(Overloaded):
        s1 = await r.open(["a"], "us", system="s", user="u", max_tokens=1)
        await r.open(["a"], "us", system="s", user="u", max_tokens=1)
    await s1.aclose()
    await (await r.open(["a"], "us", system="s", user="u", max_tokens=1)).aclose()
