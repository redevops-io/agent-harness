"""Arm ``ours-cr-v2``: ``ours-cr`` with the corrected tool set (parity tools, egress gateway, paged and recoverable output, edit_file, view_image where the endpoint is
probed to support images) and the execution supervisor (repeat/stall recovery, budget notices, deliverable checkpoint, verified-complete ending), with message
integrity checked before every model call and at most ``max_images`` images in any request. Same context policy, sampling, limits, window and judge as ``ours-cr``; nothing else differs."""
from __future__ import annotations

from . import ours
from .ours_cr import OPTIONS

NAME = "ours-cr-v2"
diagnostics = ours.diagnostics
RUN_OPTIONS = {"tools": "parity", "network": "egress", "supervise": True, "vision": "probe", "check_integrity": True,
               "max_images": 5}                    # the pinned Qwen endpoint rejects a prompt with more than 5 images (measured: HTTP 400 in the first v2 rerun)


def generate(exp, cell, workers: int = 1, run_task=None, client_factory=None, *, name: str = NAME) -> None:
    ours.generate(exp, cell, workers, run_task, client_factory, name=name, context="context-runtime", context_options=dict(OPTIONS), run_options=dict(RUN_OPTIONS))
