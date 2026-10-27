"""Arm B: ``ours-cr`` with the corrected tool set (tool parity with stock OpenCode) and NOTHING else changed: network only through the egress gateway, paged and
recoverable output, ``edit_file``, ``web_download``, truthful tool descriptions. Same context policy, sampling, limits and judge as ``ours-cr``; no recovery policy."""
from __future__ import annotations

from . import ours
from .ours_cr import OPTIONS

NAME = "ours-cr-tools"
diagnostics = ours.diagnostics
RUN_OPTIONS = {"tools": "parity", "network": "egress"}


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME) -> None:
    ours.generate(exp, cell, workers, run_task, client_factory, name=name, context="context-runtime", context_options=dict(OPTIONS), run_options=dict(RUN_OPTIONS))
