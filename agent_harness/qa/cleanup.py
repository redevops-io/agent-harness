# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verifiable cleanup.

The runner closes the session it opened on every path, including aborts. :func:`close_and_report` turns the adapter's
answer into a :class:`CleanupOutcome` whose ``signal`` is ``OK`` only if the adapter returned a ``CleanupReport`` for
this very handle with ``verified = True`` and nothing leaked. Anything else (an exception, no report, a report for
another handle, a leak) is ``FAILED`` and blocks ``PASS`` (the classifier reads it as ``CLEANUP_UNVERIFIED``).

Standard library and neutral-module imports only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .classify import Signal
from .records import CleanupReport


@dataclass(frozen=True)
class CleanupOutcome:
    signal: Signal
    report: Optional[CleanupReport] = None
    error_type: Optional[str] = None          # the exception class name only: never its (possibly hostile) message


def close_and_report(close: Callable[[Any], Any], handle: Any, handle_id: str) -> CleanupOutcome:
    """Call ``close(handle)`` and judge the report. ``handle_id`` is the id the report must carry."""
    try:
        report = close(handle)
    except Exception as exc:                                  # cleanup must never raise out of the runner
        return CleanupOutcome(Signal.FAILED, None, type(exc).__name__)
    if type(report) is not CleanupReport:
        return CleanupOutcome(Signal.FAILED, None, "NoCleanupReport")
    if report.handle_id != handle_id or not report.verified or report.leaked:
        return CleanupOutcome(Signal.FAILED, report, None)
    return CleanupOutcome(Signal.OK, report, None)


__all__ = ["CleanupOutcome", "close_and_report"]
