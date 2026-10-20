"""LLM backends behind one streaming interface, plus a router with fallback + circuit breaker.

The LLM is treated as an untrusted, unreliable vendor:
  - it only ever receives redacted text (enforced upstream, tested via FakeLLM.prompts);
  - it can be slow or down, so every call has a first-token timeout and a stall timeout;
  - fallback to the next model happens only *before the first token*. Once text has been
    streamed to the user, switching models would produce a spliced answer.

Why the google-genai SDK and not LlamaIndex's LLM wrapper: we need direct control of
timeouts and stream cancellation for backpressure. Our own fallback handles failure, so
we don't stack SDK retries on top (they would multiply latency past the request
deadline). LlamaIndex still does chunking and retrieval.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from govrag.errors import Overloaded, Unavailable


class LlmError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ChatModel(Protocol):
    name: str

    def stream(self, *, system: str, user: str, max_tokens: int) -> AsyncIterator[str]: ...


# --- fake (tests, load tests) ---------------------------------------------------------------

@dataclass
class FakeLLM:
    """Records every prompt it receives. Tests assert no raw PII ever reaches it.

    mode="echo": answers with the whole user prompt (the worst case for leaks: everything
                 in the context comes straight back to the user).
    mode="script": answers with `script(system, user)`.
    """

    name: str = "fake"
    mode: Literal["echo", "script"] = "echo"
    script: Callable[[str, str], str] | None = None
    first_token_delay_s: float = 0.0
    fail: Literal["timeout", "error", "midstream"] | None = None
    piece: int = 7
    prompts: list[tuple[str, str]] = field(default_factory=list)

    async def stream(self, *, system: str, user: str, max_tokens: int) -> AsyncIterator[str]:
        self.prompts.append((system, user))
        if self.fail == "error":
            raise LlmError("fake_error")
        if self.fail == "timeout":
            await asyncio.sleep(3600)
        if self.first_token_delay_s:
            await asyncio.sleep(self.first_token_delay_s)
        text = user if self.mode == "echo" else (self.script or (lambda s, u: ""))(system, user)
        for i in range(0, len(text), self.piece):
            if self.fail == "midstream" and i > 0:
                raise LlmError("fake_midstream")
            yield text[i : i + self.piece]
            await asyncio.sleep(0)


# --- Vertex: Gemini -------------------------------------------------------------------------

class GeminiVertex:
    def __init__(
        self, name: str, model_id: str, *, project: str, location: str, timeout_s: float
    ) -> None:
        from google import genai
        from google.genai import types

        self.name = name
        self._model = model_id
        self._types = types
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    async def stream(self, *, system: str, user: str, max_tokens: int) -> AsyncIterator[str]:
        from google.genai import errors

        try:
            it = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=user,
                config=self._types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    # Answering from given documents doesn't need reasoning tokens. On 2.5
                    # Flash they're on by default and bill as output, and they delay the
                    # first visible token.
                    thinking_config=self._types.ThinkingConfig(thinking_budget=0),
                ),
            )
            async for chunk in it:
                if chunk.text:
                    yield chunk.text
        except errors.APIError as e:
            raise LlmError(f"http_{e.code}") from e
        except (OSError, TimeoutError) as e:
            raise LlmError("connection") from e


# --- breaker + router -----------------------------------------------------------------------

@dataclass
class Breaker:
    """Opens after `failures` consecutive failures; stays open `cooldown_s`.

    Without it, a dead main model costs every request a full first-token timeout before
    the backup is tried. With it, requests skip straight to the backup during the outage.
    """

    failures: int = 3
    cooldown_s: float = 30.0
    _count: int = 0
    _open_until: float = 0.0

    def allow(self) -> bool:
        return time.monotonic() >= self._open_until

    def ok(self) -> None:
        self._count = 0

    def fail(self) -> None:
        self._count += 1
        if self._count >= self.failures:
            self._open_until = time.monotonic() + self.cooldown_s
            self._count = 0


ModelFactory = Callable[[str, str], ChatModel]  # (model name, region) -> client


class LlmRouter:
    def __init__(
        self,
        factory: ModelFactory,
        *,
        concurrency: int,
        first_token_timeout_s: float,
        stall_timeout_s: float,
        breaker_failures: int = 3,
        breaker_cooldown_s: float = 30.0,
    ) -> None:
        self._factory = factory
        self._models: dict[tuple[str, str], ChatModel] = {}
        self._breakers: dict[str, Breaker] = {}
        self._first = first_token_timeout_s
        self._stall = stall_timeout_s
        self._bf, self._bc = breaker_failures, breaker_cooldown_s
        # Caps concurrent LLM streams. Each holds an HTTP connection and a growing
        # answer buffer; past this, new requests are shed rather than queued.
        self._slots = asyncio.Semaphore(concurrency)

    def breaker(self, name: str) -> Breaker:
        return self._breakers.setdefault(name, Breaker(self._bf, self._bc))

    def _model(self, name: str, region: str) -> ChatModel:
        key = (name, region)
        if key not in self._models:
            self._models[key] = self._factory(name, region)
        return self._models[key]

    async def open(
        self,
        models: Sequence[str],
        region: str,
        *,
        system: str,
        user: str,
        max_tokens: int,
        first_token_timeout_s: float | None = None,
    ) -> LlmStream:
        """Start a stream on the first model that produces a first token.

        The caller must `await stream.aclose()` (try/finally). That returns the
        concurrency slot even if the stream is never iterated: an async generator that
        never started doesn't run its `finally`, so relying on that would leak a slot
        every time a request failed between open() and the first read.
        """
        if self._slots.locked():
            raise Overloaded("llm_busy", retry_after_s=2.0)
        await self._slots.acquire()
        first_timeout = min(self._first, first_token_timeout_s or self._first)
        tried: list[str] = []
        try:
            for name in models:
                br = self.breaker(name)
                if not br.allow():
                    continue
                tried.append(name)
                gen = self._model(name, region).stream(system=system, user=user,
                                                       max_tokens=max_tokens)
                try:
                    first: str | None = await asyncio.wait_for(anext(gen), first_timeout)
                except StopAsyncIteration:
                    first = None
                except (LlmError, TimeoutError):
                    br.fail()
                    await _close(gen)
                    continue
                br.ok()
                return LlmStream(name, first, gen, self._stall, br, self._slots.release)
        except BaseException:
            self._slots.release()
            raise
        self._slots.release()
        raise Unavailable("llm_unavailable" if tried else "llm_all_breakers_open")


class LlmStream:
    """An open model stream holding one router slot until aclose()."""

    def __init__(
        self,
        model: str,
        first: str | None,
        gen: AsyncIterator[str],
        stall_s: float,
        breaker: Breaker,
        release: Callable[[], None],
    ) -> None:
        self.model = model
        self._first = first
        self._gen = gen
        self._stall = stall_s
        self._breaker = breaker
        self._release = release
        self._closed = False

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[str]:
        try:
            if self._first is None:
                return
            yield self._first
            while True:
                try:
                    piece = await asyncio.wait_for(anext(self._gen), self._stall)
                except StopAsyncIteration:
                    return
                except TimeoutError as e:
                    self._breaker.fail()
                    raise Unavailable("llm_stalled") from e
                except LlmError as e:
                    self._breaker.fail()
                    raise Unavailable("llm_failed_midstream") from e
                yield piece
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await _close(self._gen)
        finally:
            self._release()


async def _close(gen: Any) -> None:
    aclose = getattr(gen, "aclose", None)
    if aclose is not None:
        await aclose()


def vertex_factory(
    *, project: str, timeout_s: float, model_ids: dict[str, str], locations: dict[str, str],
) -> ModelFactory:
    def make(name: str, region: str) -> ChatModel:
        if name not in model_ids:
            raise ValueError(f"no backend for model {name!r}")
        return GeminiVertex(name, model_ids[name], project=project,
                            location=locations[region], timeout_s=timeout_s)

    return make
