"""``ours-h2`` with the CORRECTED context settings, to be compared with ``ours-cr``: our harness plus the edits Harness² composed for
the task, context-runtime with task instructions pinned, placeholder-safe argument stubs (``policy="v2"``), and the loop's placeholder
guard and archive-id redirect. Differs from ``ours-cr`` only by the Harness² edits."""
from __future__ import annotations

from . import ours_h2
from .ours_cr import OPTIONS

NAME = "ours-h2-cr"
diagnostics = ours_h2.diagnostics if hasattr(ours_h2, "diagnostics") else None


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME) -> None:
    ours_h2.generate(exp, cell, workers, run_task, client_factory, name=name, context="context-runtime", context_options=dict(OPTIONS))
