"""Text -> vectors. Runs locally (ONNX), so even redacted text never leaves the box here.

A thread, not the event loop: one batch is 20 ms to 12 s of CPU. ONNX Runtime releases the
GIL while it runs, so a thread keeps the loop free. Measured (bench/loop_lag.py): during a
12 s batch, event loop lag p99 was 1.0 ms in a thread and 1.0-1.5 ms in a separate
process. No difference, so the simpler option wins. (An earlier version of this repo used
a process, based on single worst-case samples; those turned out to be macOS timer noise
that shows up even when nothing of ours runs.)

Backpressure: a semaphore caps concurrent batches (more would just fight over 6 cores and
make every request slower), and past max_pending waiting calls we shed load instead of
queueing text in memory. A timeout frees the request but not the thread, which is why the
semaphore, not the timeout, is what bounds the work.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from govrag.errors import Overloaded, Unavailable

type Vec = npt.NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    async def embed_passages(self, texts: Sequence[str]) -> list[Vec]: ...
    async def embed_query(self, text: str) -> Vec: ...


class FastEmbedder:
    def __init__(
        self, model: str, *, dim: int, concurrency: int = 2, threads: int = 2,
        max_pending: int = 64,
    ) -> None:
        """Loads the model (~6 s). Call via asyncio.to_thread at startup."""
        from fastembed import TextEmbedding

        self.dim = dim
        self._model: Any = TextEmbedding(model, threads=threads)
        list(self._model.passage_embed(["warm up"]))
        self._sem = asyncio.Semaphore(concurrency)
        self._max_pending = max_pending
        self._pending = 0

    async def _run(self, fn: Any, timeout_s: float) -> Any:
        if self._pending >= self._max_pending:
            raise Overloaded("embed_queue_full", retry_after_s=1.0)
        self._pending += 1
        try:
            async with self._sem:
                return await asyncio.wait_for(asyncio.to_thread(fn), timeout_s)
        except TimeoutError as e:
            raise Unavailable("embed_timeout") from e
        finally:
            self._pending -= 1

    async def embed_passages(self, texts: Sequence[str], timeout_s: float = 120.0) -> list[Vec]:
        if not texts:
            return []
        items = list(texts)
        out = await self._run(
            lambda: list(self._model.passage_embed(items, batch_size=64)), timeout_s
        )
        return [np.asarray(v, dtype=np.float32) for v in out]

    async def embed_query(self, text: str, timeout_s: float = 5.0) -> Vec:
        # bge models expect a query instruction prefix; query_embed adds it.
        out = await self._run(lambda: next(iter(self._model.query_embed([text]))), timeout_s)
        return np.asarray(out, dtype=np.float32)


_WORD = re.compile(r"\[[A-Z0-9_]+_[0-9a-f]{16}\]|[A-Za-z0-9]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing. For tests: no model download, ~microseconds,
    and texts sharing words (or PII tokens) really do land close together."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _one(self, text: str) -> Vec:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _WORD.findall(text):
            h = int.from_bytes(hashlib.blake2b(w.lower().encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        n = float(np.linalg.norm(v))
        if n == 0:
            v[0] = 1.0
            return v
        return v / n

    async def embed_passages(self, texts: Sequence[str]) -> list[Vec]:
        return [self._one(t) for t in texts]

    async def embed_query(self, text: str) -> Vec:
        return self._one(text)
