"""Text -> vectors. Runs locally (ONNX), so even redacted text never leaves the box here.

Why a separate process and not a thread: measured on this repo's ingest test, fastembed in
a thread still froze the event loop for 40-90 ms per batch at every batch size (4 to 64).
ONNX releases the GIL during inference, but tokenizing and post-processing around it don't.
In its own process the model has its own GIL, and the loop only waits on a future.
Cost: one extra process (~250 MB RSS) and ~1 ms of pickling per call.

The semaphore caps in-flight calls. Without it, a burst of requests queues unbounded
batches (and their text) in the executor.
"""

from __future__ import annotations

import asyncio
import hashlib
import multiprocessing as mp
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from govrag.errors import Overloaded, Unavailable

type Vec = npt.NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    async def embed_passages(self, texts: Sequence[str]) -> list[Vec]: ...
    async def embed_query(self, text: str) -> Vec: ...


_MODEL: Any = None


def _init_worker(model: str, threads: int) -> None:
    from fastembed import TextEmbedding

    global _MODEL
    _MODEL = TextEmbedding(model, threads=threads)
    list(_MODEL.passage_embed(["warm up"]))


def _passages(texts: list[str]) -> npt.NDArray[np.float32]:
    return np.stack(list(_MODEL.passage_embed(texts, batch_size=64))).astype(np.float32)


def _query(text: str) -> npt.NDArray[np.float32]:
    # bge models expect a query instruction prefix; query_embed adds it.
    return np.asarray(next(iter(_MODEL.query_embed([text]))), dtype=np.float32)


class ProcessEmbedder:
    def __init__(
        self, model: str, *, dim: int, workers: int = 1, threads: int = 2, max_pending: int = 64
    ) -> None:
        self.dim = dim
        self._args = (model, threads)
        self._workers = workers
        self._pool = self._make()
        self._max_pending = max_pending
        self._pending = 0

    def _make(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=self._workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=self._args,
        )

    async def start(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, _query, "warm")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    async def _run(self, fn: Any, arg: Any, timeout_s: float) -> Any:
        if self._pending >= self._max_pending:
            raise Overloaded("embed_queue_full", retry_after_s=1.0)
        self._pending += 1
        try:
            fut = asyncio.get_running_loop().run_in_executor(self._pool, fn, arg)
            return await asyncio.wait_for(fut, timeout_s)
        except TimeoutError as e:
            raise Unavailable("embed_timeout") from e
        except BrokenProcessPool as e:
            old, self._pool = self._pool, self._make()
            old.shutdown(wait=False, cancel_futures=True)
            raise Unavailable("embed_worker_died") from e
        finally:
            self._pending -= 1

    async def embed_passages(self, texts: Sequence[str], timeout_s: float = 120.0) -> list[Vec]:
        if not texts:
            return []
        arr = await self._run(_passages, list(texts), timeout_s)
        return list(arr)

    async def embed_query(self, text: str, timeout_s: float = 5.0) -> Vec:
        out: Vec = await self._run(_query, text, timeout_s)
        return out


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
