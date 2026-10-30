# SPDX-License-Identifier: AGPL-3.0-or-later
"""The production :class:`~agent_harness.qa.ports.Clock`: wall-clock UTC for custody fields, a monotonic counter for
deadlines. Everything else in the QA module receives a ``Clock`` as an argument; this is the one place that reads real
time, so every other module stays deterministic and tests substitute a fake clock.

Standard library only.
"""
from __future__ import annotations

import datetime
import time


class SystemClock:
    """``Clock`` backed by the operating system."""

    def now_utc(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def monotonic_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


__all__ = ["SystemClock"]
