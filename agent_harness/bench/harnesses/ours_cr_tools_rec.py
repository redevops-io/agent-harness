"""Arm C: arm B plus the progress-aware recovery policy (``agent_harness.progress``): guidance, classification and telemetry, with the HARD STOP DISABLED
(the overall turn and time limits still apply). Nothing else differs from arm B."""
from __future__ import annotations

from . import ours
from .ours_cr import OPTIONS
from .ours_cr_tools import RUN_OPTIONS as TOOLS

NAME = "ours-cr-tools-rec"
diagnostics = ours.diagnostics
RUN_OPTIONS = {**TOOLS, "recovery": True, "recovery_hard_stop": False}


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME) -> None:
    ours.generate(exp, cell, workers, run_task, client_factory, name=name, context="context-runtime", context_options=dict(OPTIONS), run_options=dict(RUN_OPTIONS))
