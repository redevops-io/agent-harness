"""Context ablation arm: our harness with context-runtime's age-based compaction, in the CORRECTED form: task instructions pinned,
placeholder-safe argument stubs (``policy="v2"``), the placeholder guard and archive-id redirect (always on in the loop and tools).
Differs from ``ours-cr-pin`` only in not protecting small input reads."""
from __future__ import annotations

from . import ours

NAME = "ours-cr"
diagnostics = ours.diagnostics
OPTIONS = {"policy": "v2"}


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME) -> None:
    ours.generate(exp, cell, workers, run_task, client_factory, name=name, context="context-runtime", context_options=dict(OPTIONS))
