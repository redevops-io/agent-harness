"""``ours-h2`` WITHOUT context-runtime: our harness plus the Harness²-composed edits, with no proactive context management.
Paired with ``ours-h2`` it measures what context-runtime costs or buys when the refined harness is in use."""
from __future__ import annotations

from . import ours_h2

NAME = "ours-h2-nocr"


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None) -> None:
    ours_h2.generate(exp, cell, workers, run_task, client_factory, name=NAME, context="none")
