"""Picklable helpers for tests that run code in worker processes."""

from __future__ import annotations

import time


def sleep_then(seconds: float) -> float:
    time.sleep(seconds)
    return seconds
